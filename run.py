#!/usr/bin/env python3
"""Export one S1 label copy and full original-episode S1/S2/(S3)/S4 timelines."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import csv
import fcntl
import gzip
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pyarrow.parquet as pq

import pipeline as p

DEFAULT_SOURCE=Path('/mnt/pfs/datasets/Processed/Egostandard_stageA_260915_delay1_trunc/stage_a_aligned')
DEFAULT_OUTPUT=Path('/mnt/pfs/Data/ryk/egostandard/runs/labels_annotations_20260928')


def prepare_label_metadata(source,output,rows,info,partial):
    root=output/'labels_s1'
    meta=root/'meta'
    epdir=meta/'episodes/chunk-000'
    epdir.mkdir(parents=True,exist_ok=True)
    target=epdir/'file-000.parquet'
    pq.write_table(rows,target.with_suffix('.parquet.tmp'),compression='zstd')
    target.with_suffix('.parquet.tmp').replace(target)
    for name in ('tasks.jsonl','tasks.parquet'):
        if (source/'meta'/name).exists(): shutil.copy2(source/'meta'/name,meta/name)
    mapping=source/'meta/episode_source_map.jsonl'
    if mapping.exists():
        if not partial: shutil.copy2(mapping,meta/mapping.name)
        else:
            with mapping.open() as inp,(meta/mapping.name).open('w') as out:
                out.writelines(itertools.islice(inp,len(rows)))
    fixed=dict(info)
    fixed['features']=dict(info['features'])
    for prefix in ('observation.state','action'):
        for field,dtype in (('valid','bool'),('repaired','bool'),('anomaly_reason','uint8')):
            fixed['features'][f'{prefix}.eef_{field}']={'dtype':dtype,'shape':[2],'names':['left','right'],'fps':30}
    fixed['features']['observation.state.hand_features_valid']={'dtype':'bool','shape':[2],'names':['left','right'],'fps':30}
    fixed.update(total_episodes=len(rows),total_frames=int(np.asarray(rows['length']).sum()),
                 artifact_type='labels_only',source_dataset=str(source),source_video_root=str(source),
                 statistics_status='not exported; source EEF statistics are not repaired-label statistics',
                 trajectory_cleaning={'method':p.RULE_CONFIG['S1'],'source_modified':False,
                                      'frame_numbering':'source episode/frame numbering retained',
                                      'partial_test_output':partial})
    p.write_json(meta/'info.json',fixed)
    p.write_json(meta/'statistics_policy.json',{
        'status':'not_computed','note':'No original EEF statistics copied as repaired statistics. Recompute when building training normalization.',
        'use_masks':['observation.state.eef_valid','action.eef_valid'],'use_timeline':True})


def ordered_records(pool,source,output,rows,skip,checkpoint,reuse_stream,resume,workers,batch_size):
    def jobs():
        for batch_id,table in enumerate(rows.to_batches(max_chunksize=batch_size)):
            block=table.to_pylist()
            reuse=None
            if reuse_stream is not None:
                reuse=[]
                for _ in block:
                    line=reuse_stream.readline()
                    if not line: raise ValueError('S1 manifest ended before source metadata')
                    reuse.append(json.loads(line))
            yield (str(source),str(output),block,skip,str(checkpoint/f'batch-{batch_id:05d}.jsonl'),reuse,resume)
        if reuse_stream is not None and reuse_stream.readline(): raise ValueError('S1 manifest has extra episodes')
    it=iter(jobs())
    pending=[pool.submit(p.process_batch,j) for j in itertools.islice(it,workers*2)]
    while pending:
        yield from pending.pop(0).result()
        try:pending.append(pool.submit(p.process_batch,next(it)))
        except StopIteration:pass


def export(args):
    source,output=args.source.resolve(),args.output.resolve()
    if source==output or source in output.parents or output in source.parents:
        raise ValueError('Output must be independent of source')
    if not source.is_dir(): raise FileNotFoundError(source)
    rows,info,metadata_ids=p.load_metadata(source)
    if args.limit:rows=rows.slice(0,min(args.limit,len(rows)))
    initial=int(np.asarray(rows['length']).sum())
    output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        return export_locked(args,source,output,rows,info,metadata_ids,initial)
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def export_locked(args,source,output,rows,info,metadata_ids,initial):
    fingerprint={'version':p.VERSION,'source':str(source),'source_metadata_identity':metadata_ids,
                 'planned_episodes':len(rows),'planned_frames':initial,'partial_test_output':bool(args.limit),
                 'code_sha256':p.code_hashes(),'rules':p.RULE_CONFIG,
                 'label_policy':'same rows and original dtypes; only audited EEF repairs plus quality flags',
                 'video_policy':'reference original files only; no open/write/link/cut/encode operations'}
    config_path=output/'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text())!=fingerprint:
            raise ValueError('Source/code/config fingerprint differs. Use a new output directory.')
    else:
        if (output/'labels_s1').exists() or (output/'annotations').exists():
            raise ValueError('Untracked label/annotation directories already exist')
        p.write_json(config_path,fingerprint)
    variant='without_s3' if args.skip_s3 else 'with_s3'
    dest=output/'annotations'/variant
    complete_path=dest/'COMPLETE.json'
    if complete_path.exists():
        done=json.loads(complete_path.read_text())
        if done['status']!='complete':raise ValueError('Invalid completion marker')
        print(json.dumps({'status':'already_complete','variant':variant,'summary':str(dest/'summary.json'),
                          'labels':str(output/'labels_s1'),'timeline':str(dest/'timeline.csv.gz')}),flush=True)
        return done
    if dest.exists() and not args.resume:
        raise ValueError('Incomplete variant exists. Use --resume to continue verified checkpoints.')
    dest.mkdir(parents=True,exist_ok=True)
    checkpoint=dest/'checkpoints'
    checkpoint.mkdir(exist_ok=True)
    label_root=output/'labels_s1'
    label_complete=label_root/'COMPLETE.json'
    reuse=label_complete.exists()
    if reuse:
        ls=json.loads(label_complete.read_text())
        if ls['processed_episodes']!=len(rows) or ls['processed_frames']!=initial:
            raise ValueError('S1 completion totals differ from requested source')
        if p.sha256(label_root/'manifest.jsonl')!=ls['manifest_sha256']:
            raise ValueError('S1 manifest checksum mismatch')
    else:
        # Prevent switching S3 modes halfway through an unfinished S1 export.
        other=output/'annotations'/('with_s3' if args.skip_s3 else 'without_s3')
        if other.exists():raise ValueError('Finish/resume the original variant before reusing S1')
        prepare_label_metadata(source,output,rows,info,bool(args.limit))
    run={'status':'running','variant':variant,'stage_order':list(p.stages_for(args.skip_s3)),
         's3_enabled':not args.skip_s3,'s5_enabled':False,'source':str(source),'output':str(output),
         'label_reuse':reuse,'started_at_utc':p.now(),'planned_episodes':len(rows),'planned_frames':initial,
         'workers':args.workers,'batch_size':args.batch_size,'video_files_opened':0,'video_files_written':0}
    p.write_json(dest/'run.json',run)
    started=time.monotonic()
    stages={s:Counter() for s in p.stages_for(args.skip_s3)}
    repairs=Counter();totals=Counter();reason_frames=Counter()
    plan_digest=hashlib.sha256();s1_digest=hashlib.sha256()
    filenames=('plan.jsonl','timeline.csv.gz','rejected_intervals.csv.gz','episodes.csv.gz')
    with ExitStack() as stack:
        plan=stack.enter_context((dest/'plan.jsonl.tmp').open('w'))
        timeline=stack.enter_context(gzip.open(dest/'timeline.csv.gz.tmp','wt',newline='',compresslevel=1))
        rejected=stack.enter_context(gzip.open(dest/'rejected_intervals.csv.gz.tmp','wt',newline='',compresslevel=1))
        episodes=stack.enter_context(gzip.open(dest/'episodes.csv.gz.tmp','wt',newline='',compresslevel=1))
        tw,rw,ew=csv.writer(timeline),csv.writer(rejected),csv.writer(episodes)
        timeline_fields=('source_episode_index','start_frame','end_frame_exclusive','start_seconds','end_seconds',
                         'duration_seconds','status','stage','reason','reference_mp4_start_seconds','reference_mp4_end_seconds',
                         'aux_mp4_start_seconds','aux_mp4_end_seconds')
        tw.writerow(timeline_fields);rw.writerow(timeline_fields)
        ew.writerow(('source_episode_index','source_frames','duration_seconds','repaired_label_path',
                     'reference_video_path','reference_from_timestamp','reference_to_timestamp',
                     'aux_video_path','aux_from_timestamp','aux_to_timestamp','rejected_frames','accepted_frames'))
        s1_manifest=None
        reused_manifest=None
        if reuse:reused_manifest=stack.enter_context((label_root/'manifest.jsonl').open())
        else:s1_manifest=stack.enter_context((label_root/'manifest.jsonl.tmp').open('w'))
        pool=stack.enter_context(ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('fork')))
        for i,rec in enumerate(ordered_records(pool,source,output,rows,args.skip_s3,checkpoint,
                                              reused_manifest,args.resume,args.workers,args.batch_size),1):
            if rec['source_episode_index']!=i-1:raise AssertionError('Ordered output episode mismatch')
            line=json.dumps(rec,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
            plan.write(line);plan_digest.update(line.encode())
            if s1_manifest is not None:
                label_rec={k:v for k,v in rec.items() if k not in ('stages','accepted_intervals','timeline','videos')}
                label_line=json.dumps(label_rec,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
                s1_manifest.write(label_line);s1_digest.update(label_line.encode())
            n=rec['source_length']
            invalid=sum(rec['stages'][s]['removed_frames'] for s in stages)
            accepted=sum(b-a for a,b in rec['accepted_intervals'])
            if invalid+accepted!=n:raise AssertionError('Episode frame conservation')
            totals.update(episodes=1,source_frames=n,rejected_frames=invalid,accepted_frames=accepted,
                          accepted_segments=len(rec['accepted_intervals']),affected_episodes=int(invalid>0),
                          timeline_rows=len(rec['timeline']),label_bytes=rec['repaired_label_identity'][2])
            repairs.update(rec['repair_counts'])
            for s,stat in rec['stages'].items():
                stages[s].update({k:v for k,v in stat.items() if k!='removed_intervals'})
                stages[s]['removed_intervals']+=len(stat['removed_intervals'])
            ref,aux=rec['videos']['camera_reference'],rec['videos']['camera_aux_0']
            ew.writerow((rec['source_episode_index'],n,f'{n/30:.6f}',str(label_root/rec['label_relative_path']),
                         ref['path'],ref['from_timestamp'],ref['to_timestamp'],
                         aux['path'],aux['from_timestamp'],aux['to_timestamp'],invalid,accepted))
            cursor=0
            for a,b,code,reason in rec['timeline']:
                if a!=cursor or not a<b<=n:raise AssertionError('Timeline gap or overlap')
                cursor=b
                row=(rec['source_episode_index'],a,b,f'{a/30:.6f}',f'{b/30:.6f}',f'{(b-a)/30:.6f}',
                     'reject' if code else 'keep',f'S{code}' if code else '',p.REASONS[reason],
                     f'{ref["from_timestamp"]+a/30:.6f}',f'{ref["from_timestamp"]+b/30:.6f}',
                     f'{aux["from_timestamp"]+a/30:.6f}',f'{aux["from_timestamp"]+b/30:.6f}')
                tw.writerow(row)
                if code:
                    rw.writerow(row);totals['rejected_table_rows']+=1
                    reason_frames[p.REASONS[reason]]+=b-a
            if cursor!=n:raise AssertionError('Timeline incomplete')
            if i%1000==0 or i==len(rows):
                plan.flush()
                if s1_manifest is not None:s1_manifest.flush()
                progress=dict(run,processed_episodes=i,elapsed_seconds=round(time.monotonic()-started,1),
                              stage_removed_frames={s:stages[s]['removed_frames'] for s in stages})
                p.write_json(dest/'progress.json',progress)
                print(json.dumps({k:progress[k] for k in ('status','variant','processed_episodes','planned_episodes',
                                                        'elapsed_seconds','stage_removed_frames')}),flush=True)
    if totals['episodes']!=len(rows) or totals['source_frames']!=initial:raise AssertionError('Source coverage')
    if totals['accepted_frames']+totals['rejected_frames']!=initial:raise AssertionError('Full frame conservation')
    if any(p.identity(source/rel)!=ident for rel,ident in metadata_ids.items()):raise RuntimeError('Source metadata changed')
    for name in filenames:(dest/(name+'.tmp')).replace(dest/name)
    if not reuse:
        (label_root/'manifest.jsonl.tmp').replace(label_root/'manifest.jsonl')
        p.write_json(label_complete,{'status':'complete','version':p.VERSION,'source':str(source),
            'processed_episodes':totals['episodes'],'processed_frames':initial,'label_bytes':totals['label_bytes'],
            'manifest_sha256':s1_digest.hexdigest(),'repair_counts':dict(repairs),
            'partial_test_output':bool(args.limit),'completed_at_utc':p.now(),
            'source_modified':False,'video_files_opened':0,'video_files_written':0})
    stage_summary={s:dict(v,removed_seconds=v['removed_frames']/30) for s,v in stages.items()}
    checksums={name:p.sha256(dest/name) for name in filenames}
    if checksums['plan.jsonl']!=plan_digest.hexdigest():raise AssertionError('Plan checksum mismatch')
    summary=dict(run,status='complete',completed_at_utc=p.now(),elapsed_seconds=time.monotonic()-started,
                 processed_episodes=totals['episodes'],processed_frames=initial,stages=stage_summary,
                 repair_counts=dict(repairs),counts=dict(totals),total_removed_frames=totals['rejected_frames'],
                 total_removed_seconds=totals['rejected_frames']/30,removed_share=totals['rejected_frames']/initial,
                 accepted_frames=totals['accepted_frames'],accepted_seconds=totals['accepted_frames']/30,
                 label_bytes=totals['label_bytes'],reason_removed_frames=dict(reason_frames),
                 plan_sha256=checksums['plan.jsonl'],checksums=checksums,partial_test_output=bool(args.limit),
                 rule_config=p.RULE_CONFIG,label_path=str(label_root),timeline_path=str(dest/'timeline.csv.gz'),
                 timeline_basis='original source episode frames, [start,end); source MP4 offsets also recorded')
    p.write_json(dest/'summary.json',summary)
    p.write_json(dest/'COMPLETE.json',summary)
    p.write_json(output/'LATEST.json',{'status':'complete','variant':variant,'summary':str(dest/'summary.json'),
                                     'labels':str(label_root),'timeline':str(dest/'timeline.csv.gz')})
    print(json.dumps({'status':'complete','variant':variant,'episodes':totals['episodes'],
                      'label_bytes':totals['label_bytes'],'total_removed_frames':totals['rejected_frames'],
                      'stage_removed_frames':{s:v['removed_frames'] for s,v in stages.items()},
                      'timeline':str(dest/'timeline.csv.gz')}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT,help='Output work directory under Data/ryk by default')
    parser.add_argument('--workers',type=int,default=min(64,os.cpu_count() or 1))
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--skip-s3',action='store_true',help='Run S1 -> S2 -> S4; reuse completed S1 labels')
    parser.add_argument('--resume',action='store_true',help='Continue an incomplete matching variant from checked batch checkpoints')
    parser.add_argument('--limit',type=int,default=0,help='Test only: first N episodes, 0 means full dataset')
    args=parser.parse_args()
    if min(args.workers,args.batch_size)<1 or args.limit<0:parser.error('Invalid workers/batch-size/limit')
    if sys.flags.optimize:parser.error('Do not use Python -O; validation assertions are required')
    try:export(args)
    except Exception as exc:
        output=args.output.resolve()
        if (output/'config.json').is_file():
            # Preserve checkpoints; a failure never creates COMPLETE.json.
            p.write_json(output/'LAST_ERROR.json',{'status':'failed','at_utc':p.now(),'error':f'{type(exc).__name__}: {exc}',
                                                  'variant':'without_s3' if args.skip_s3 else 'with_s3'})
        raise


if __name__=='__main__':
    main()
