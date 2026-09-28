"""Full-dataset camera motion annotation: endpoint motion over exactly 0.2 s."""
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
from scipy.spatial.transform import Rotation

ROOT=Path('/mnt/pfs/datasets/Processed/Egostandard_stageA_260915_delay1_trunc/stage_a_aligned')
FPS=30
EXT='extrinsic.camera_reference.T_Episode_CameraReference'
CAM='observation.images.camera_reference'
VP='videos/'+CAM+'/'
FIELDS=['frame_index','timestamp','episode_index',EXT]
KINDS=('rotation','translation')
CONFIG=dict(rule='camera_motion_over_0.2s',version=1,fps=FPS,window_seconds=.2,window_frame_gap=6,
            rotation_threshold_deg=30.,translation_threshold_m=.30,comparison='strictly greater than',
            rotation_measure='Endpoint relative-rotation geodesic angle, not cumulative angle.',
            translation_measure='Endpoint camera-center Euclidean displacement in fixed episode coordinates, in meters; not cumulative path.',
            smoothing=False,window_stride_frames=1,merge='Union of overlapping or touching hit windows, separately for each kind; never bridge a positive-duration gap.',
            interval_convention='start_frame and end_frame are inclusive pose endpoints; time span equals (end_frame-start_frame)/30; clip_end_frame_exclusive=end_frame+1 includes the final evidence image.',
            invalid_data='Non-finite, missing, invalid transforms or inconsistent timestamps are episode errors, not motion hits.',
            label='camera_motion_threshold_hit',semantic_quality_judgment=False)

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
    columns = ['episode_index', 'length', 'data/chunk_index', 'data/file_index']
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


def read_episode(root,row):
    path=Path(root)/f'data/chunk-{row["data/chunk_index"]:03d}/file-{row["data/file_index"]:03d}.parquet'
    before=file_identity(path);table=pq.read_table(path,columns=FIELDS,use_threads=False);n=len(table)
    assert n==row['length'] and n>0,'Episode length mismatch'
    assert np.array_equal(array(table,'frame_index',()),np.arange(n)),'Frame indices mismatch'
    assert np.all(array(table,'episode_index',())==row['episode_index']),'Episode indices mismatch'
    assert np.allclose(array(table,'timestamp',()),np.arange(n)/FPS,atol=.001,rtol=0),'Timestamp mismatch'
    assert abs(row[VP+'to_timestamp']-row[VP+'from_timestamp']-n/FPS)<.001,'Video duration mismatch'
    ext=array(table,EXT,(4,4)).astype(np.float64)
    assert np.allclose(ext[:,3],[0,0,0,1],atol=1e-5,rtol=0),'Invalid homogeneous transform'
    rot=ext[:,:3,:3]
    assert np.max(np.abs(rot.transpose(0,2,1)@rot-np.eye(3)))<=1e-3,'Non-orthogonal camera rotation'
    assert np.all(np.linalg.det(rot)>0),'Camera rotation reflection'
    assert file_identity(path)==before,'Source changed during reading'
    return ext


def motion_metrics(ext):
    gap=CONFIG['window_frame_gap'];n=len(ext)
    if n<=gap:return np.empty(0),np.empty(0)
    rotations=Rotation.from_matrix(ext[:,:3,:3])
    angular=np.rad2deg((rotations[:-gap].inv()*rotations[gap:]).magnitude())
    position=ext[:,:3,3]
    distance=np.linalg.norm(position[gap:]-position[:-gap],axis=1)
    return angular,distance


def hit_indices(angles,distances):
    return dict(rotation=np.flatnonzero(angles>CONFIG['rotation_threshold_deg']),
                translation=np.flatnonzero(distances>CONFIG['translation_threshold_m']))


def consecutive_runs(indices):
    if not len(indices):return []
    split=np.flatnonzero(np.diff(indices)>1)+1
    return [[int(group[0]),int(group[-1])+1] for group in np.split(indices,split)]


def make_segments(indices,angles,distances,kind,offset=0.):
    if not len(indices):return []
    gap=CONFIG['window_frame_gap'];groups=np.split(indices,np.flatnonzero(np.diff(indices)>gap)+1)
    values=angles if kind=='rotation' else distances
    segments=[]
    for number,group in enumerate(groups,1):
        start=int(group[0]);end=int(group[-1])+gap;peak=int(group[np.argmax(values[group])])
        segments.append(dict(kind=kind,segment_index=number,start_frame=start,end_frame=end,clip_end_frame_exclusive=end+1,
                             start_time_seconds=round(offset+start/FPS,6),end_time_seconds=round(offset+end/FPS,6),
                             duration_seconds=round((end-start)/FPS,6),hit_window_count=len(group),hit_window_start_runs=consecutive_runs(group),
                             max_rotation_deg=float(angles[group].max()),max_translation_cm=float(distances[group].max()*100),
                             peak_window_start_frame=peak,peak_window_end_frame=peak+gap,
                             peak_start_time_seconds=round(offset+peak/FPS,6),peak_end_time_seconds=round(offset+(peak+gap)/FPS,6),
                             peak_rotation_deg=float(angles[peak]),peak_translation_cm=float(distances[peak]*100)))
    return segments


def process(job):
    root,row=job;ep=int(row['episode_index']);n=int(row['length'])
    try:
        ext=read_episode(root,row);angles,distances=motion_metrics(ext);hits=hit_indices(angles,distances)
        offset=float(row[VP+'from_timestamp'])
        intervals={k:make_segments(hits[k],angles,distances,k,offset) for k in KINDS}
        coverage={k:np.zeros(max(n-1,0),bool) for k in KINDS}
        for k in KINDS:
            for span in intervals[k]:coverage[k][span['start_frame']:span['end_frame']]=True
        counts={k+'_hit_windows':len(hits[k]) for k in KINDS}
        counts.update({k+'_covered_frame_intervals':int(coverage[k].sum()) for k in KINDS})
        counts['both_same_window_hits']=int(np.intersect1d(hits['rotation'],hits['translation']).size)
        counts['overlap_covered_frame_intervals']=int((coverage['rotation']&coverage['translation']).sum())
        counts['union_covered_frame_intervals']=int((coverage['rotation']|coverage['translation']).sum())
        result=dict(status='ok',episode_index=ep,frames=n,evaluated_windows=len(angles),counts=counts)
        if any(intervals.values()):
            video=Path(root)/f'videos/{CAM}/chunk-{row[VP+"chunk_index"]:03d}/file-{row[VP+"file_index"]:03d}.mp4'
            assert video.is_file(),'Missing source video'
            result['video']=dict(episode_index=ep,video_path=str(video),label=CONFIG['label'],video_duration_seconds=round(n/FPS,6),
                                video_start_timestamp_seconds=offset,rotation_segment_count=len(intervals['rotation']),
                                translation_segment_count=len(intervals['translation']),rotation_hit_window_count=len(hits['rotation']),
                                translation_hit_window_count=len(hits['translation']),max_rotation_deg=float(angles.max()),
                                max_translation_cm=float(distances.max()*100),
                                rotation_duration_seconds=round(counts['rotation_covered_frame_intervals']/FPS,6),
                                translation_duration_seconds=round(counts['translation_covered_frame_intervals']/FPS,6),
                                overlap_duration_seconds=round(counts['overlap_covered_frame_intervals']/FPS,6),
                                union_duration_seconds=round(counts['union_covered_frame_intervals']/FPS,6),
                                rotation_intervals=intervals['rotation'],translation_intervals=intervals['translation'])
            result['windows']=[dict(kind=k,start_frame=int(i),end_frame=int(i)+6,start_time_seconds=round(offset+int(i)/FPS,6),
                                    end_time_seconds=round(offset+(int(i)+6)/FPS,6),rotation_deg=float(angles[i]),translation_cm=float(distances[i]*100)) for k in KINDS for i in hits[k]]
        return result
    except Exception as exc:return dict(status='error',episode_index=ep,frames=n,evaluated_windows=max(n-6,0),error=f'{type(exc).__name__}: {exc}')


def batch(jobs):return [process(job) for job in jobs]


def bounded(pool,root,table,workers):
    source=([(str(root),r) for r in t.to_pylist()] for t in table.to_batches(max_chunksize=32))
    pending=[pool.submit(batch,j) for j in itertools.islice(source,workers*4)]
    while pending:
        yield from pending.pop(0).result()
        try:job=next(source)
        except StopIteration:continue
        pending.append(pool.submit(batch,job))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=32)
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    assert root!=out and root not in out.parents;out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();table,files,info=load_metadata(root)
    meta_identity={str(p):file_identity(p) for p in files+[root/'meta/info.json']}
    expected_windows=sum(max(int(n)-6,0) for n in np.asarray(table['length']))
    json_write(out/'config.json',dict(**CONFIG,source=str(root)))
    vc=['episode_index','video_path','label','video_duration_seconds','rotation_segment_count','translation_segment_count','rotation_hit_window_count','translation_hit_window_count','max_rotation_deg','max_translation_cm','rotation_duration_seconds','translation_duration_seconds','overlap_duration_seconds','union_duration_seconds','rotation_intervals_json','translation_intervals_json']
    sc=['episode_index','video_path','kind','segment_index','start_time_seconds','end_time_seconds','duration_seconds','start_frame','end_frame','clip_end_frame_exclusive','hit_window_count','max_rotation_deg','max_translation_cm','peak_start_time_seconds','peak_end_time_seconds','peak_rotation_deg','peak_translation_cm']
    wc=['episode_index','video_path','kind','start_frame','end_frame','start_time_seconds','end_time_seconds','rotation_deg','translation_cm']
    counts=Counter();last=0
    from contextlib import ExitStack
    with ExitStack() as stack:
        jf=stack.enter_context((out/'motion_videos.jsonl').open('x'));ef=stack.enter_context((out/'errors.jsonl').open('x'))
        handles=[jf,ef];writers={}
        for filename,fields in [('motion_videos',vc),('rotation_videos',vc),('translation_videos',vc),('rotation_segments',sc),('translation_segments',sc),('motion_windows',wc)]:
            f=stack.enter_context((out/(filename+'.csv')).open('x',encoding='utf-8-sig',newline=''));handles.append(f)
            writers[filename]=csv.DictWriter(f,fieldnames=fields);writers[filename].writeheader()
        pool=stack.enter_context(ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('fork')))
        for result in bounded(pool,root,table,args.workers):
            counts['processed_episodes']+=1;counts['processed_frames']+=result['frames'];counts['evaluated_windows']+=result['evaluated_windows']
            if result['status']=='error':counts['error_episodes']+=1;ef.write(json.dumps(result)+'\n')
            else:
                counts['successful_episodes']+=1;counts.update(result['counts'])
                if 'video' in result:
                    v=result['video'];counts['hit_videos']+=1;jf.write(json.dumps(v)+'\n')
                    cr={k:v[k] for k in vc if k in v}
                    for k in KINDS:cr[k+'_intervals_json']=json.dumps(v[k+'_intervals'],separators=(',',':'))
                    writers['motion_videos'].writerow(cr)
                    if all(v[k+'_segment_count'] for k in KINDS):counts['both_type_videos']+=1
                    for k in KINDS:
                        if not v[k+'_segment_count']:continue
                        counts[k+'_videos']+=1;counts[k+'_segments']+=v[k+'_segment_count'];writers[k+'_videos'].writerow(cr)
                        for span in v[k+'_intervals']:
                            sr={key:span[key] for key in sc if key in span};sr.update(episode_index=v['episode_index'],video_path=v['video_path']);writers[k+'_segments'].writerow(sr)
                    for window in result['windows']:writers['motion_windows'].writerow(dict(episode_index=v['episode_index'],video_path=v['video_path'],**window))
            elapsed=time.perf_counter()-started
            if elapsed-last>=25:
                last=elapsed
                for f in handles:f.flush()
                progress=dict(counts,expected_episodes=len(table),elapsed_seconds=round(elapsed,2));json_write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    assert counts['processed_episodes']==len(table) and counts['processed_frames']==info['total_frames'] and counts['evaluated_windows']==expected_windows
    assert all(file_identity(Path(p))==x for p,x in meta_identity.items()),'Source metadata changed'
    for key in ['hit_videos','both_type_videos','rotation_videos','translation_videos','rotation_segments','translation_segments','error_episodes']:counts.setdefault(key,0)
    summary=dict(status='scan_complete' if not counts['error_episodes'] else 'scan_complete_with_errors',source=str(root),output=str(out),
                 expected_episodes=len(table),expected_frames=info['total_frames'],expected_windows=expected_windows,**dict(counts),
                 rotation_duration_seconds=round(counts['rotation_covered_frame_intervals']/FPS,6),
                 translation_duration_seconds=round(counts['translation_covered_frame_intervals']/FPS,6),
                 overlap_duration_seconds=round(counts['overlap_covered_frame_intervals']/FPS,6),
                 union_duration_seconds=round(counts['union_covered_frame_intervals']/FPS,6),
                 elapsed_seconds=round(time.perf_counter()-started,3),workers=args.workers,config=CONFIG,
                 source_preservation='Read-only source access, each parquet identity stable during reading and all metadata identities unchanged.')
    json_write(out/'summary.json',summary);json_write(out/'progress.json',dict(summary,status='complete'))
    json_write(out/'SHA256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()

