"""Audit v2 exports against independent detector masks on repaired labels."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow.parquet as pq
import pipeline as p


def span_mask(spans,n):
    mask=np.zeros(n,bool);end=0
    for a,b in spans:
        assert end<=a<b<=n and not mask[a:b].any()
        mask[a:b]=True;end=b
    return mask


def oracle(table,row,rec,skip,thresholds):
    """Build masks directly from detector primitives, without annotate()."""
    n=len(table)
    valid=p.column(table,'observation.state.eef_valid',2).astype(bool)
    s1=~valid.all(axis=1)
    if any(d['external_target_unresolved'] for d in rec['repair_details'].values()):s1[-1]=True
    assert np.array_equal(s1,span_mask(rec['s1_removed_intervals'],n))
    ext=p.edge.array(table,p.EXT,(4,4)).astype(float)
    position=np.stack([p.column(table,f'observation.state.eef.{side}.position',3)
                       for side in p.SIDES],axis=1).astype(float)
    s2=np.zeros(n,bool)
    for a,b in p.runs(~s1):
        projected={side:p.edge.project(position[a:b,h],ext[a:b],row['camera_intrinsics']['camera_reference'])
                   for h,side in enumerate(p.SIDES)}
        for hit in p.edge.identify(projected,row['camera_intrinsics']['camera_reference']):
            s2[a+hit['start_frame']:a+hit['end_frame_exclusive']]=True
    common=p.runs(~(s1|s2))
    s3,s4=np.zeros(n,np.uint8),np.zeros(n,np.uint8)
    for a,b in common:
        got=p.carry_detect(ext[a:b],position[a:b],np.ones((b-a,2),bool),p.S4_CONFIG)
        for hit in got['matching_window_runs']:
            s4[a+hit['start']:a+hit['end']] |= 1 if hit['rule'].endswith('_translation') else 2
        if not skip:
            angles,distance=p.motion.motion_metrics(ext[a:b])
            for mask,bit in ((distance>thresholds['translation_threshold_m'],1),
                             (angles>thresholds['rotation_threshold_deg'],2)):
                for start in np.flatnonzero(mask):s3[a+start:a+start+7] |= bit
    codes,reasons=np.zeros(n,np.uint8),np.zeros(n,np.uint8)
    codes[s1],reasons[s1]=1,1
    codes[s2],reasons[s2]=2,2
    codes[s4!=0],reasons[s4!=0]=4,40+s4[s4!=0]
    owned=(s3!=0)&(codes==0)
    codes[owned],reasons[owned]=3,30+s3[owned]
    return codes,reasons,{'S1':s1,'S2':s2,'S4':s4!=0,'S3':s3!=0},s3,s4


def audit(output,s1_reference=None):
    output=Path(output)
    config=json.loads((output/'config.json').read_text())
    source=Path(config['source'])
    rows,_,ids=p.load_metadata(source)
    assert ids==config['source_metadata_identity']
    labels=output/'labels_s1'
    complete=json.loads((labels/'COMPLETE.json').read_text())
    n=complete['processed_episodes'];rows=rows.slice(0,n).to_pylist()
    manifest=[json.loads(line) for line in (labels/'manifest.jsonl').open()]
    assert len(manifest)==n and p.sha256(labels/'manifest.jsonl')==complete['manifest_sha256']
    assert len(list((labels/'data').glob('chunk-*/file-*.parquet')))==n
    reference=[json.loads(line) for line in Path(s1_reference).open()] if s1_reference else None
    if reference is not None:assert len(reference)==n
    for i,rec in enumerate(manifest):
        assert rec['source_episode_index']==i
        path=labels/rec['label_relative_path']
        assert p.identity(path)==rec['repaired_label_identity']
        assert p.sha256(path)==rec['repaired_label_sha256']
        assert p.identity(source/rec['label_relative_path'])==rec['source_data_identity']
        if reference is not None:
            assert rec['repaired_label_sha256']==reference[i]['repaired_label_sha256']
    report={'status':'passed','episodes':n,'frames':complete['processed_frames'],
            'label_files':n,'label_bytes':complete['label_bytes'],'source_modified':False,
            's1_byte_identical_to_previous':reference is not None,
            'checks':['label hashes and source identities','independent detector oracle',
                      'union and overlap conservation','all rule branches retained',
                      'full CSV coverage and values','S4 invariant under S3 switch','single S1 label copy']}
    prior=None
    for variant in ('with_s3','without_s3'):
        root=output/'annotations'/variant
        summary=json.loads((root/'summary.json').read_text());skip=variant=='without_s3'
        assert summary['status']=='complete' and summary['processed_episodes']==n
        assert summary['annotation_policy']==p.ANNOTATION_POLICY and summary['timeline_schema_version']==2
        for name,digest in summary['checksums'].items():assert p.sha256(root/name)==digest
        records=[json.loads(line) for line in (root/'plan.jsonl').open()]
        assert len(records)==n
        removed,matched=Counter(),Counter();rejected,overlap,table_rows=0,0,0
        with gzip.open(root/'timeline.csv.gz','rt',newline='') as stream:
            iterator=iter(csv.DictReader(stream))
            for ep,(rec,row,manifest_rec) in enumerate(zip(records,rows,manifest)):
                assert rec['source_episode_index']==row['episode_index']==ep
                assert rec['repaired_label_identity']==manifest_rec['repaired_label_identity']
                assert rec['source_data_identity']==manifest_rec['source_data_identity']
                assert set(rec['stages'])==set(p.stages_for(skip))
                table=pq.read_table(labels/rec['label_relative_path'],columns=p.FILTER_FIELDS,use_threads=False)
                p.validate_source_rows(table,row)
                codes,reasons,masks,s3,s4=oracle(table,row,rec,skip,summary['rule_config']['S3'])
                length=rec['source_length']
                assert length==len(table)
                assert rec['accepted_intervals']==[[a,b] for a,b in p.runs(codes==0)]
                assert rec['s3_s4_overlap_frames']==int(((s3!=0)&(s4!=0)).sum())
                overlap+=rec['s3_s4_overlap_frames'];rejected+=int((codes!=0).sum())
                for stage,stat in rec['stages'].items():
                    assert np.array_equal(span_mask(stat['matched_intervals'],length),masks[stage])
                    assert np.array_equal(span_mask(stat['removed_intervals'],length),codes==int(stage[1]))
                    assert stat['matched_frames']==int(masks[stage].sum())
                    assert stat['removed_frames']==int((codes==int(stage[1])).sum())
                    removed[stage]+=stat['removed_frames'];matched[stage]+=stat['matched_frames']
                if not skip:
                    assert rec['stages']['S3']['detection_input_frames']==rec['stages']['S4']['detection_input_frames']
                    assert rec['stages']['S3']['detection_input_segments']==rec['stages']['S4']['detection_input_segments']
                if prior is not None:assert rec['stages']['S4']==prior[ep]['stages']['S4']
                cursor=0
                assert len(rec['timeline'])==len(rec['rule_timeline'])
                for (a,b,stage,reason),(ma,mb,matches) in zip(rec['timeline'],rec['rule_timeline']):
                    assert a==cursor and a<b<=length and (a,b)==(ma,mb);cursor=b
                    assert np.all(codes[a:b]==stage) and np.all(reasons[a:b]==reason)
                    expected=([stage] if stage in (1,2) else
                              ([40+int(s4[a])] if s4[a] else [])+([30+int(s3[a])] if s3[a] else []))
                    assert matches==expected
                    if stage not in (1,2):
                        assert np.all(s3[a:b]==s3[a]) and np.all(s4[a:b]==s4[a])
                    csv_row=next(iterator)
                    assert int(csv_row['source_episode_index'])==ep
                    assert int(csv_row['start_frame'])==a and int(csv_row['end_frame_exclusive'])==b
                    assert csv_row['status']==('reject' if stage else 'keep')
                    assert csv_row['stage']==(f'S{stage}' if stage else '') and csv_row['reason']==p.REASONS[reason]
                    assert csv_row['matched_stages']==';'.join(p.reason_stage(match) for match in matches)
                    assert csv_row['matched_reasons']==';'.join(p.REASONS[match] for match in matches)
                    for field,value in (('start_seconds',a/30),('end_seconds',b/30),('duration_seconds',(b-a)/30)):
                        assert abs(float(csv_row[field])-value)<1e-6
                    for view,prefix in (('camera_reference','reference'),('camera_aux_0','aux')):
                        offset=rec['videos'][view]['from_timestamp']
                        assert abs(float(csv_row[f'{prefix}_mp4_start_seconds'])-offset-a/30)<1e-6
                        assert abs(float(csv_row[f'{prefix}_mp4_end_seconds'])-offset-b/30)<1e-6
                    table_rows+=1
                assert cursor==length
            assert next(iterator,None) is None
        for stage,stat in summary['stages'].items():
            assert removed[stage]==stat['removed_frames'] and matched[stage]==stat['matched_frames']
        assert rejected==summary['total_removed_frames'] and overlap==summary['s3_s4_overlap_frames']
        assert sum(matched.values())-overlap==rejected==sum(removed.values())
        assert table_rows==summary['counts']['timeline_rows']
        report[variant]={'stage_removed_frames':dict(removed),'stage_matched_frames':dict(matched),
                         'total_removed_frames':rejected,'s3_s4_overlap_frames':overlap,
                         'timeline_rows':table_rows,'label_reuse':summary['label_reuse']}
        prior=records
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--s1-reference',type=Path,help='Optional previous S1 manifest for byte-identical label checks')
    args=parser.parse_args()
    report=audit(args.output,args.s1_reference)
    p.write_json(args.output/'INTEGRATION_AUDIT.json',report)
    print(json.dumps(report))
