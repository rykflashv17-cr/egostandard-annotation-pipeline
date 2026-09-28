"""Audit completed test outputs against both previously approved full plans."""
import argparse
from collections import Counter
import csv
import gzip
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pipeline as p


def audit(output,with_reference,without_reference):
    labels=output/'labels_s1'
    complete=json.loads((labels/'COMPLETE.json').read_text())
    n=complete['processed_episodes']
    manifest=[json.loads(line) for line in (labels/'manifest.jsonl').open()]
    assert len(manifest)==n
    assert p.sha256(labels/'manifest.jsonl')==complete['manifest_sha256']
    assert len(list((labels/'data').glob('chunk-*/file-*.parquet')))==n
    identities={r['source_episode_index']:r['repaired_label_identity'] for r in manifest}
    for rec in manifest:
        path=labels/rec['label_relative_path']
        assert p.identity(path)==rec['repaired_label_identity']
        assert p.sha256(path)==rec['repaired_label_sha256']
    report={'status':'passed','episodes':n,'frames':complete['processed_frames'],
            'label_files':n,'label_bytes':complete['label_bytes'],'source_modified':False,
            'checks':['label hashes','single label copy','source file identities','full timeline coverage',
                      'stage intervals match previous full plans','S3 skipped and S4 recomputed']}
    for variant,reference in (('with_s3',with_reference),('without_s3',without_reference)):
        root=output/'annotations'/variant
        summary=json.loads((root/'summary.json').read_text())
        assert summary['status']=='complete' and summary['processed_episodes']==n
        for name,digest in summary['checksums'].items():assert p.sha256(root/name)==digest
        changed=Counter();compared=0
        with (root/'plan.jsonl').open() as current,reference.open() as previous:
            for line,old in zip(current,itertools.islice(previous,n)):
                rec,base=json.loads(line),json.loads(old)
                assert rec['source_episode_index']==base['source_episode_index']==compared
                assert rec['source_data_identity']==base['source_data_identity']
                assert p.identity(Path(summary['source'])/rec['label_relative_path'])==rec['source_data_identity']
                assert identities[compared]==rec['repaired_label_identity']
                assert ('S3' in rec['stages'])==(variant=='with_s3')
                for stage in rec['stages']:
                    if rec['stages'][stage]['removed_intervals']!=base['stages'][stage]['removed_intervals']:
                        changed[stage]+=1
                cursor=0
                for a,b,stage,reason in rec['timeline']:
                    assert a==cursor and a<b<=rec['source_length'];cursor=b
                    assert reason in p.REASONS
                assert cursor==rec['source_length']
                compared+=1
        assert compared==n
        if changed:raise AssertionError(f'{variant}: baseline interval differences: {dict(changed)}')
        with gzip.open(root/'timeline.csv.gz','rt',newline='') as stream:
            rows=sum(1 for _ in csv.DictReader(stream))
        assert rows==summary['counts']['timeline_rows']
        report[variant]={'stage_removed_frames':{s:v['removed_frames'] for s,v in summary['stages'].items()},
                         'total_removed_frames':summary['total_removed_frames'],'timeline_rows':rows,
                         'baseline_interval_differences':dict(changed),'label_reuse':summary['label_reuse']}
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--with-reference',type=Path,required=True)
    parser.add_argument('--without-reference',type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.output,args.with_reference,args.without_reference)
    p.write_json(args.output/'INTEGRATION_AUDIT.json',report)
    print(json.dumps(report))
