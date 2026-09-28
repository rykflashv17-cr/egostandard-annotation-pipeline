"""Check S4 invariance and final-set monotonicity on previously reviewed episodes."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow.parquet as pq
import pipeline as p

FIELDS=['frame_index','episode_index','timestamp','index',p.EXT]
FIELDS += [f'{prefix}.eef.{side}.{field}' for prefix in ('observation.state','action')
           for side in p.SIDES for field in ('position','rotation_6d')]
FIELDS += [f'{prefix}.gripper.{side}.closedness' for prefix in ('observation.state','action') for side in p.SIDES]


def rejected(result,n):
    mask=np.zeros(n,bool)
    for a,b,stage,_ in result['timeline']:
        if stage:mask[a:b]=True
    return mask


def audit(source,cases):
    rows,_,_=p.load_metadata(source)
    selected=json.loads(Path(cases).read_text())['selected']
    results=[];original=(p.S3_TRANSLATION_M,p.S3_ROTATION_DEG)
    try:
        for case in selected:
            row=rows.slice(case['source_episode_index'],1).to_pylist()[0]
            path=Path(source)/p.data_relative(row)
            identity=p.identity(path)
            assert identity==case['source_data_identity'] and row['length']==case['source_length']
            table=pq.read_table(path,columns=FIELDS,use_threads=False)
            cleaned,s1,_,_=p.s1_repair(table,row)
            p.configure_s3(.15,15)
            a=p.annotate_details(cleaned,row,s1)
            skipped=p.annotate_details(cleaned,row,s1,True)
            p.configure_s3(.20,20)
            b=p.annotate_details(cleaned,row,s1)
            assert a['stages']['S4']==b['stages']['S4']==skipped['stages']['S4']
            am,bm,sm=(rejected(x,len(table)) for x in (a,b,skipped))
            assert not np.any(bm&~am)
            assert not np.any(sm&~bm)
            assert p.identity(path)==identity
            results.append({'source_episode_index':row['episode_index'],
                            'frames':len(table),'A_rejected_frames':int(am.sum()),
                            'B_rejected_frames':int(bm.sum()),'skip_rejected_frames':int(sm.sum()),
                            's4_matched_frames':a['stages']['S4']['matched_frames']})
    finally:p.configure_s3(*original)
    return {'status':'passed','episodes':len(results),'source_frames':sum(r['frames'] for r in results),
            's4_invariant_across_A_B_and_skip':True,'B_rejection_subset_of_A':True,
            'skip_rejection_subset_of_B':True,'source_modified':False,'label_files_written':0,
            'video_files_opened':0,'video_files_written':0,'cases':results}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.source,args.cases)
    p.write_json(args.report,result)
    print(json.dumps(result))
