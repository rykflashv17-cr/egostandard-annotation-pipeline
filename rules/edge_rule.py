"""Rule 1: BOTH end effectors simultaneously at 5% ego edges for >=2s."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq

ROOT=Path('/mnt/pfs/datasets/Processed/Egostandard_stageA_260915_delay1_trunc/stage_a_aligned')
FPS=30
EXT='extrinsic.camera_reference.T_Episode_CameraReference'
CAM='observation.images.camera_reference'
VP='videos/'+CAM+'/'
SIDES=('left','right')
FIELDS=['frame_index','timestamp','episode_index',EXT]+[f'observation.state.eef.{s}.position' for s in SIDES]
CONFIG=dict(rule='rule_01_bilateral_edge',version=1,fps=FPS,edge_fraction=.05,
            minimum_duration_seconds_inclusive=2.0,min_bad_frames=60,
            include_out_of_frame=False,behind_camera_counts_as_out_of_frame=True,
            simultaneous_both_hands=True,require_same_border=False,
            activity_filter=False,smoothing=False,merge_bad_gaps=False,
            interval_convention='[start_frame/30 + video_offset, end_frame_exclusive/30 + video_offset)',
            pose_source='observation.state.eef.{left,right}.position',
            edge_definition='Inside the image, with x<0.05W or x>=0.95W or y<0.05H or y>=0.95H.',
            projection='Invert per-frame T_Episode_CameraReference, then use stored-size intrinsics and distortion.',
            label='unqualified_bilateral_edge',
            interpretation='Geometric end-effector projection rule, not direct visual hand segmentation; known projection discrepancies may produce false positives.')

def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))



def project(pos, extrinsic, intr):
    pos = np.asarray(pos, dtype=np.float64)
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    xyz = np.einsum('nji,nj->ni', extrinsic[:, :3, :3], pos-extrinsic[:, :3, 3])
    K = np.asarray(intr['matrix'], dtype=np.float64)
    dist = np.asarray(intr['dist_coeffs'], dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsics')
    if not np.isfinite(dist).all():
        raise ValueError('Invalid camera distortion')
    forward = xyz[:, 2] > 1e-6
    uv = np.full((len(pos), 2), np.nan)
    if forward.any():
        if np.any(dist != 0):
            import cv2
            uv[forward] = cv2.projectPoints(xyz[forward], np.zeros(3), np.zeros(3), K, dist)[0].reshape(-1, 2)
        else:
            homogeneous = xyz[forward] @ K.T
            uv[forward] = homogeneous[:, :2]/homogeneous[:, 2:]
    width, height = (int(intr['stored_image_size'][k]) for k in ('width', 'height'))
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image size')
    inside = forward & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    m = CONFIG['edge_fraction']
    safe = inside & (uv[:, 0] >= m*width) & (uv[:, 0] < (1-m)*width) & (uv[:, 1] >= m*height) & (uv[:, 1] < (1-m)*height)
    return uv, inside & ~safe, ~inside, ~forward



def array(table, key, shape):
    values = table[key].combine_chunks()
    if values.null_count:
        raise ValueError(f'Null values in {key}')
    for _ in shape:
        values = values.flatten()
        if values.null_count:
            raise ValueError(f'Nested null values in {key}')
    out = values.to_numpy(zero_copy_only=False).reshape((len(table),)+shape)
    if not np.isfinite(out).all():
        raise ValueError(f'Non-finite values in {key}')
    return out



def file_identity(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]



def json_write(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)



def load_metadata(root):
    files = sorted((root/'meta/episodes').glob('chunk-*/file-*.parquet'))
    if not files:
        raise ValueError('No episode metadata')
    columns = ['episode_index', 'length', 'data/chunk_index', 'data/file_index', 'camera_intrinsics']
    columns += [VP+k for k in ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp')]
    import pyarrow as pa
    table = pa.concat_tables([pq.read_table(p, columns=columns, use_threads=False) for p in files])
    table = table.take(np.argsort(np.asarray(table['episode_index'])))
    episodes = np.asarray(table['episode_index'])
    if not np.array_equal(episodes, np.arange(len(table))):
        raise ValueError('Unexpected episode numbering')
    pairs = np.stack([np.asarray(table[VP+'chunk_index']), np.asarray(table[VP+'file_index'])], axis=1)
    if len(np.unique(pairs, axis=0)) != len(table):
        raise ValueError('Shared video files require an explicit aggregation policy')
    info = json.loads((root/'meta/info.json').read_text())
    if info['fps'] != FPS or info['total_episodes'] != len(table):
        raise ValueError('Dataset info and metadata differ')
    if int(np.asarray(table['length']).sum()) != info['total_frames']:
        raise ValueError('Total frame count differs from info')
    return table, files, info


def identify(projections, intr, offset=0.0):
    masks = {side:(p[1] | p[2] if CONFIG['include_out_of_frame'] else p[1]) for side,p in projections.items()}
    result=[]
    width,height=[int(intr['stored_image_size'][k]) for k in ('width','height')]
    for start,end in runs(masks['left'] & masks['right']):
        if end-start<CONFIG['min_bad_frames']:continue
        stats={}
        for side,(uv,edge,out,behind) in projections.items():
            coords=uv[start:end];finite=np.isfinite(coords).all(axis=1)
            stats[side]=dict(edge_frames=int(edge[start:end].sum()),out_of_frame_frames=int(out[start:end].sum()),
                             behind_camera_frames=int(behind[start:end].sum()),
                             normalized_uv_min=(coords[finite]/[width,height]).min(axis=0).tolist() if finite.any() else None,
                             normalized_uv_max=(coords[finite]/[width,height]).max(axis=0).tolist() if finite.any() else None)
        result.append(dict(start_frame=start,end_frame_exclusive=end,start_time_seconds=round(offset+start/FPS,6),
                           end_time_seconds=round(offset+end/FPS,6),duration_seconds=round((end-start)/FPS,6),
                           hand='both',reason='bilateral_edge_or_out_of_frame' if CONFIG['include_out_of_frame'] else 'bilateral_in_frame_edge',hand_stats=stats))
    return result


def prepare_episode(root,row):
    path=Path(root)/f'data/chunk-{row["data/chunk_index"]:03d}/file-{row["data/file_index"]:03d}.parquet'
    before=file_identity(path)
    table=pq.read_table(path,columns=FIELDS,use_threads=False);n=len(table)
    assert n==row['length'] and n>0,'Episode length mismatch'
    assert np.array_equal(array(table,'frame_index',()),np.arange(n)),'Frame index mismatch'
    assert np.all(array(table,'episode_index',())==row['episode_index']),'Episode index mismatch'
    assert np.allclose(array(table,'timestamp',()),np.arange(n)/FPS,atol=.001,rtol=0),'Timestamp mismatch'
    assert abs(row[VP+'to_timestamp']-row[VP+'from_timestamp']-n/FPS)<.001,'Video duration mismatch'
    ext=array(table,EXT,(4,4))
    assert np.allclose(ext[:,3],np.array([0,0,0,1]),atol=1e-5,rtol=0),'Homogeneous transform invalid'
    rot=ext[:,:3,:3]
    assert np.max(np.abs(np.swapaxes(rot,1,2)@rot-np.eye(3)))<=1e-3,'Camera rotation invalid'
    assert np.all(np.linalg.det(rot)>0),'Camera reflection'
    intr=row['camera_intrinsics']['camera_reference']
    projections={s:project(array(table,f'observation.state.eef.{s}.position',(3,)),ext,intr) for s in SIDES}
    assert file_identity(path)==before,'Source changed during reading'
    return projections,intr,n


def process(job):
    root,row=job;ep=int(row['episode_index']);n=int(row['length'])
    try:
        projections,intr,n=prepare_episode(root,row)
        intervals=identify(projections,intr,float(row[VP+'from_timestamp']))
        left,right=projections['left'],projections['right']
        counts=dict(both_in_frame_edge_frames=int((left[1]&right[1]).sum()),
                    both_edge_or_out_frames=int(((left[1]|left[2])&(right[1]|right[2])).sum()),
                    bad_frames=sum(s['end_frame_exclusive']-s['start_frame'] for s in intervals))
        result=dict(status='ok',episode_index=ep,frames=n,**counts)
        if intervals:
            video=Path(root)/f'videos/{CAM}/chunk-{row[VP+"chunk_index"]:03d}/file-{row[VP+"file_index"]:03d}.mp4'
            assert video.is_file(),'Missing source video'
            result['video']=dict(episode_index=ep,video_path=str(video),label=CONFIG['label'],
                                video_duration_seconds=round(n/FPS,6),video_start_timestamp_seconds=float(row[VP+'from_timestamp']),
                                bad_segment_count=len(intervals),bad_duration_seconds=round(counts['bad_frames']/FPS,6),
                                bad_ratio=round(counts['bad_frames']/n,6),bad_intervals=intervals)
        return result
    except Exception as exc:return dict(status='error',episode_index=ep,frames=n,error=f'{type(exc).__name__}: {exc}')


def process_batch(jobs):return [process(j) for j in jobs]


def bounded(pool,root,table,workers):
    batches=([(str(root),r) for r in batch.to_pylist()] for batch in table.to_batches(max_chunksize=32))
    pending=[pool.submit(process_batch,j) for j in itertools.islice(batches,workers*4)]
    while pending:
        yield from pending.pop(0).result()
        try:job=next(batches)
        except StopIteration:continue
        pending.append(pool.submit(process_batch,job))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=32)
    parser.add_argument('--sample',type=int,default=0);parser.add_argument('--include-out-of-frame',action='store_true')
    args=parser.parse_args();CONFIG['include_out_of_frame']=args.include_out_of_frame
    root=args.root.resolve();out=args.output.resolve();assert root!=out and root not in out.parents
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();table,files,_=load_metadata(root)
    meta_identity={str(p):file_identity(p) for p in files+[root/'meta/info.json']}
    if args.sample:table=table.take(sorted(np.random.default_rng(2026091703).choice(len(table),min(args.sample,len(table)),replace=False)))
    expected_episodes=len(table);expected_frames=int(np.asarray(table['length']).sum())
    json_write(out/'config.json',dict(**CONFIG,source=str(root),sample=bool(args.sample)))
    vc=['episode_index','video_path','label','video_duration_seconds','bad_segment_count','bad_duration_seconds','bad_ratio','bad_intervals_json']
    sc=['episode_index','video_path','start_time_seconds','end_time_seconds','duration_seconds','start_frame','end_frame_exclusive','left_edge_frames','left_out_of_frame_frames','right_edge_frames','right_out_of_frame_frames']
    counts=Counter();last_update=0
    with (out/'bad_videos.jsonl').open('x') as jf,(out/'bad_videos.csv').open('x',encoding='utf-8-sig',newline='') as vf,(out/'bad_segments.csv').open('x',encoding='utf-8-sig',newline='') as sf,(out/'errors.jsonl').open('x') as ef,ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('fork')) as pool:
        vw=csv.DictWriter(vf,fieldnames=vc);vw.writeheader();sw=csv.DictWriter(sf,fieldnames=sc);sw.writeheader()
        for result in bounded(pool,root,table,args.workers):
            counts['processed_episodes']+=1;counts['processed_frames']+=result['frames']
            if result['status']=='error':counts['error_episodes']+=1;ef.write(json.dumps(result)+'\n')
            else:
                counts['successful_episodes']+=1
                for key in ['both_in_frame_edge_frames','both_edge_or_out_frames','bad_frames']:counts[key]+=result[key]
                if 'video' in result:
                    row=result['video'];counts['bad_videos']+=1;counts['bad_segments']+=row['bad_segment_count']
                    jf.write(json.dumps(row)+'\n')
                    csvrow={k:row[k] for k in vc if k in row};csvrow['bad_intervals_json']=json.dumps(row['bad_intervals'],separators=(',',':'));vw.writerow(csvrow)
                    for span in row['bad_intervals']:
                        sr={k:span[k] for k in sc if k in span};sr.update(episode_index=row['episode_index'],video_path=row['video_path'])
                        for side in SIDES:
                            for kind in ['edge_frames','out_of_frame_frames']:sr[side+'_'+kind]=span['hand_stats'][side][kind]
                        sw.writerow(sr)
            elapsed=time.perf_counter()-started
            if elapsed-last_update>=25:
                last_update=elapsed
                for f in (jf,vf,sf,ef):f.flush()
                progress=dict(counts,expected_episodes=expected_episodes,elapsed_seconds=round(elapsed,2));json_write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    assert counts['processed_episodes']==expected_episodes and counts['processed_frames']==expected_frames
    assert all(file_identity(Path(p))==identity for p,identity in meta_identity.items()),'Source metadata changed'
    for k in ['bad_videos','bad_segments','bad_frames','error_episodes']:counts.setdefault(k,0)
    summary=dict(status='scan_complete' if not counts['error_episodes'] else 'scan_complete_with_errors',source=str(root),output=str(out),
                 sample=bool(args.sample),expected_episodes=expected_episodes,expected_frames=expected_frames,**dict(counts),
                 bad_video_ratio=counts['bad_videos']/expected_episodes,bad_duration_seconds=round(counts['bad_frames']/FPS,6),
                 elapsed_seconds=round(time.perf_counter()-started,3),workers=args.workers,config=CONFIG,
                 source_preservation='Read-only source access; each parquet identity stable across read; metadata identities unchanged.')
    json_write(out/'summary.json',summary);json_write(out/'progress.json',dict(summary,status='complete'))
    json_write(out/'SHA256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

