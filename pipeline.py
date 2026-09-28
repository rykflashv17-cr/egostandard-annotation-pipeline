"""S1 repaired labels and sequential S1/S2/(S3)/S4 original-video annotations."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'rules'))
from detector import Config as DetectorConfig, detect_hand, runs
from repair_core import repair_table
from scan import column
import edge_rule as edge
import camera_rule as motion
from cosine_rule import Config as CarryConfig, detect as carry_detect

VERSION = 'egostandard-labels-and-timelines-v1'
FPS = 30
SIDES = ('left', 'right')
VIEWS = ('camera_reference', 'camera_aux_0')
EXT = 'extrinsic.camera_reference.T_Episode_CameraReference'
S3_TRANSLATION_M, S3_ROTATION_DEG = .075, 7.5
S4_CONFIG = CarryConfig(window_seconds=1, translation_m=.20, rotation_deg=20.,
                        cosine_p95_p05=.03, sustained_hit_seconds=1)
edge.CONFIG['include_out_of_frame'] = True
RULE_CONFIG = {
    'S1': {'detector': DetectorConfig().json(), 'max_gap_frames': 30,
           'position': 'linear interpolation', 'rotation': 'shortest-path SO(3) SLERP',
           'unresolved': 'keep source rows/values; mark invalid and reject their video intervals',
           'action_alignment': 'delay1 retained; no cut-tail clamping or row deletion'},
    'S2': dict(edge.CONFIG),
    'S3': {'window_seconds': .2, 'window_frame_gap': 6,
           'translation_threshold_m': S3_TRANSLATION_M, 'rotation_threshold_deg': S3_ROTATION_DEG,
           'comparison': 'strictly greater than', 'interval': 'include both endpoint images and all between'},
    'S4': S4_CONFIG.json(),
}
REASONS = {0: '', 1: 'S1_unrepairable_trajectory_or_terminal_action',
           2: 'S2_bilateral_edge_or_out_of_frame',
           31: 'S3_camera_translation', 32: 'S3_camera_rotation', 33: 'S3_translation_and_rotation',
           41: 'S4_stable_hands_camera_translation', 42: 'S4_stable_hands_camera_rotation',
           43: 'S4_stable_hands_translation_and_rotation'}
FILTER_FIELDS = ['frame_index', 'episode_index', 'timestamp', EXT,
                 'observation.state.eef_valid', 'action.eef_valid']
FILTER_FIELDS += [f'observation.state.eef.{s}.position' for s in SIDES]


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temp.replace(path)


def code_hashes():
    files = [ROOT/'pipeline.py', ROOT/'run.py'] + sorted((ROOT/'rules').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def data_relative(row):
    return f'data/chunk-{int(row["data/chunk_index"]):03d}/file-{int(row["data/file_index"]):03d}.parquet'


def video_records(source, row):
    n = int(row['length'])
    result = {}
    for view in VIEWS:
        p = f'videos/observation.images.{view}/'
        start, end = float(row[p+'from_timestamp']), float(row[p+'to_timestamp'])
        if abs(end-start-n/FPS) > .001:
            raise ValueError(('Video metadata duration mismatch', row['episode_index'], view))
        path = Path(source) / f'{p}chunk-{int(row[p+"chunk_index"]):03d}/file-{int(row[p+"file_index"]):03d}.mp4'
        result[view] = {'path': str(path), 'from_timestamp': start, 'to_timestamp': end}
    return result


def load_metadata(source):
    source = Path(source)
    files = sorted((source/'meta/episodes').glob('chunk-*/file-*.parquet'))
    if not files:
        raise ValueError('No source episode metadata')
    rows = pa.concat_tables([pq.read_table(p, use_threads=False) for p in files])
    rows = rows.take(np.argsort(np.asarray(rows['episode_index'])))
    info = json.loads((source/'meta/info.json').read_text())
    if info['fps'] != FPS or info['total_episodes'] != len(rows):
        raise ValueError('Source must be 30 FPS EgoStandard with consistent episode count')
    if not np.array_equal(np.asarray(rows['episode_index']), np.arange(len(rows))):
        raise ValueError('Source episodes must be contiguous and ordered')
    if int(np.asarray(rows['length']).sum()) != info['total_frames']:
        raise ValueError('Source frame total mismatch')
    pairs = np.stack([np.asarray(rows[f'data/{x}_index']) for x in ('chunk', 'file')], axis=1)
    if len(np.unique(pairs, axis=0)) != len(rows):
        raise ValueError('This entry point requires one source Parquet per episode')
    for view in VIEWS:
        p = f'videos/observation.images.{view}/'
        pairs = np.stack([np.asarray(rows[p+x+'_index']) for x in ('chunk', 'file')], axis=1)
        if len(np.unique(pairs, axis=0)) != len(rows):
            raise ValueError('This entry point requires independent source episode videos')
    return rows, info, {str(p.relative_to(source)): identity(p) for p in files+[source/'meta/info.json']}


def merge(spans):
    result = []
    for a,b in sorted((int(a), int(b)) for a,b in spans if a < b):
        if result and a <= result[-1][1]:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([a,b])
    return result


def subtract(kept, removed):
    removed = merge(removed)
    result = []
    for a,b in kept:
        cursor = a
        for x,y in removed:
            if y <= cursor: continue
            if x >= b: break
            if cursor < x: result.append([cursor, min(x,b)])
            cursor = max(cursor, min(y,b))
        if cursor < b: result.append([cursor,b])
    return result


def validate_source_rows(table, row):
    n,ep = int(row['length']), int(row['episode_index'])
    if n < 1 or len(table) != n or not np.array_equal(column(table,'frame_index'), np.arange(n)):
        raise ValueError(('Source frame indexing mismatch', ep))
    if not np.all(column(table,'episode_index') == ep):
        raise ValueError(('Source episode indexing mismatch', ep))
    if not np.allclose(column(table,'timestamp'), np.arange(n)/FPS, atol=.001, rtol=0):
        raise ValueError(('Source timestamp mismatch', ep))
    if 'dataset_from_index' in row and 'index' in table.column_names:
        if not np.array_equal(column(table,'index'), np.arange(row['dataset_from_index'],row['dataset_to_index'])):
            raise ValueError(('Source global indexing mismatch',ep))


def s1_repair(table, row):
    """Detect on original float64 poses; write repaired values in their original dtype."""
    validate_source_rows(table, row)
    if 'observation.state.eef_valid' in table.column_names:
        raise ValueError('Expected original labels, not an already repaired input')
    n = len(table)
    record = {'frames': n, 'episode_index': int(row['episode_index']), 'hands': {}}
    codes = {}
    config = DetectorConfig()
    for side in SIDES:
        sp,sr,ap,ar = [column(table,f'{prefix}.eef.{side}.{field}', width).astype(np.float64)
                      for prefix in ('observation.state','action')
                      for field,width in (('position',3),('rotation_6d',6))]
        detected = detect_hand(sp,sr,ap,ar,config)
        sg,ag = [column(table,f'{prefix}.gripper.{side}.closedness',1) for prefix in ('observation.state','action')]
        gripper_mismatch = int((~np.isclose(ag[:-1,0],sg[1:,0],atol=config.delay_atol,rtol=0,equal_nan=True)).sum())
        if detected['delay_mismatch_rows'] or gripper_mismatch:
            raise ValueError(('Unexpected EEF/gripper delay1 mismatch',row['episode_index'],side))
        codes['state_'+side+'_reason'] = detected['state']
        codes['action_'+side+'_reason'] = detected['action']
        record['hands'][side] = {
            'eef_delay_mismatch_rows': detected['delay_mismatch_rows'],
            'gripper_delay_mismatch_rows': gripper_mismatch,
            'terminal_action': detected['terminal_action'],
            'external_terminal_target_code': detected['external_terminal_target_code'],
        }
    cleaned, details, flags = repair_table(table, record, codes, 30)
    bad = ~flags['state_valid'].all(axis=1)
    external_bad = any(d['external_target_unresolved'] for d in details.values())
    if external_bad: bad[-1] = True
    counts = {
        'repaired_hand_frames': int(flags['state_repaired'].sum()),
        'unresolved_hand_frames': int((~flags['state_valid']).sum()),
        'unresolved_hand_intervals': sum(len(d['unresolved_intervals']) for d in details.values()),
        'repaired_action_hand_rows': int(flags['action_repaired'].sum()),
        'unresolved_action_hand_rows': int((~flags['action_valid']).sum()),
    }
    if external_bad: counts['external_terminal_rows_removed'] = 1
    changed = {f'{prefix}.eef.{side}.{field}' for prefix in ('observation.state','action')
               for side in SIDES for field in ('position','rotation_6d')}
    for name in table.column_names:
        if name not in changed and not arrays_equal(cleaned[name],table[name]):
            raise AssertionError(('Unmodified field changed', name))
    # Repair core preserves all unmarked values, including raw keypoints/grippers.
    for kind,prefix in (('state','observation.state'),('action','action')):
        for hand,side in enumerate(SIDES):
            for field,width in (('position',3),('rotation_6d',6)):
                name=f'{prefix}.eef.{side}.{field}'
                mask=~flags[kind+'_repaired'][:,hand]
                if not np.array_equal(column(table,name,width)[mask],column(cleaned,name,width)[mask],equal_nan=True):
                    raise AssertionError(('Unmarked EEF value changed',name))
    return cleaned, [[a,b] for a,b in runs(bad)], counts, details


def arrays_equal(left,right):
    if left.equals(right): return True
    left,right=left.combine_chunks(),right.combine_chunks()
    if left.type != right.type or left.null_count != right.null_count:
        return False
    while pa.types.is_list(left.type) or pa.types.is_large_list(left.type):
        if not left.offsets.equals(right.offsets) or not left.is_null().equals(right.is_null()): return False
        left,right=left.flatten(),right.flatten()
    if pa.types.is_floating(left.type):
        return np.array_equal(left.to_numpy(zero_copy_only=False),right.to_numpy(zero_copy_only=False),equal_nan=True)
    return left.equals(right)


def write_label(table,target):
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    temp=target.with_suffix('.parquet.tmp')
    pq.write_table(table,temp,compression='zstd',compression_level=1)
    saved=pq.read_table(temp,use_threads=False)
    if not table.schema.equals(saved.schema,check_metadata=True) or len(table)!=len(saved):
        raise AssertionError(('Output label schema/length mismatch',target))
    if any(not arrays_equal(table[name],saved[name]) for name in table.column_names):
        raise AssertionError(('Output label roundtrip mismatch',target))
    temp.replace(target)
    return identity(target), sha256(target)


def stages_for(skip_s3):
    return ('S1','S2','S4') if skip_s3 else ('S1','S2','S3','S4')


def detect_s2(kept,position,ext,intr):
    bad=[]
    for a,b in kept:
        projected={side:edge.project(position[a:b,h],ext[a:b],intr) for h,side in enumerate(SIDES)}
        bad.extend([a+hit['start_frame'],a+hit['end_frame_exclusive']] for hit in edge.identify(projected,intr))
    return merge(bad)


def detect_s3(kept,ext,n):
    bits=np.zeros(n,np.uint8)
    for a,b in kept:
        angles,dist=motion.motion_metrics(ext[a:b])
        for kind,mask,bit in (('translation',dist>S3_TRANSLATION_M,1),('rotation',angles>S3_ROTATION_DEG,2)):
            for hit in motion.make_segments(np.flatnonzero(mask),angles,dist,kind):
                bits[a+hit['start_frame']:a+hit['clip_end_frame_exclusive']] |= bit
    return [[a,b] for a,b in runs(bits!=0)],bits


def detect_s4(kept,position,ext,n):
    bits=np.zeros(n,np.uint8)
    for a,b in kept:
        got=carry_detect(ext[a:b],position[a:b],np.ones((b-a,2),bool),S4_CONFIG)
        for hit in got['matching_window_runs']:
            bit=1 if hit['rule'].endswith('_translation') else 2
            bits[a+hit['start']:a+hit['end']] |= bit
    return [[a,b] for a,b in runs(bits!=0)],bits


def annotate(table,row,s1,skip_s3=False):
    validate_source_rows(table,row)
    n=len(table)
    ext=edge.array(table,EXT,(4,4)).astype(np.float64)
    rotation=ext[:,:3,:3]
    if (not np.allclose(ext[:,3],[0,0,0,1],atol=1e-5,rtol=0)
        or np.max(np.abs(rotation.transpose(0,2,1)@rotation-np.eye(3)))>1e-3
        or np.any(np.linalg.det(rotation)<=0)):
        raise ValueError('Camera transforms are not rigid')
    position=np.stack([column(table,f'observation.state.eef.{side}.position',3) for side in SIDES],axis=1).astype(np.float64)
    valid=column(table,'observation.state.eef_valid',2).astype(bool)
    state_bad=~valid.all(axis=1)
    expected=np.zeros(n,bool)
    for a,b in s1: expected[a:b]=True
    if np.any(state_bad & ~expected): raise AssertionError('S1 misses invalid state rows')
    codes=np.zeros(n,np.uint8)
    reason=np.zeros(n,np.uint8)
    kept=[[0,n]]
    stages={}
    for stage in stages_for(skip_s3):
        if stage=='S1': removed=s1; branch=None
        elif stage=='S2': removed=detect_s2(kept,position,ext,row['camera_intrinsics']['camera_reference']);branch=None
        elif stage=='S3': removed,branch=detect_s3(kept,ext,n)
        else: removed,branch=detect_s4(kept,position,ext,n)
        before=sum(b-a for a,b in kept)
        next_kept=subtract(kept,removed)
        after=sum(b-a for a,b in next_kept)
        if before-after!=sum(b-a for a,b in removed): raise AssertionError((stage,'frame conservation'))
        number=int(stage[1:])
        for a,b in removed:
            if not 0<=a<b<=n or np.any(codes[a:b]): raise AssertionError((stage,'overlap or bounds'))
            codes[a:b]=number
            reason[a:b]=number if branch is None else number*10+branch[a:b]
        stages[stage]={'input_segments':len(kept),'input_frames':before,'removed_intervals':removed,
                       'removed_frames':before-after,'kept_segments':len(next_kept),'kept_frames':after}
        kept=next_kept
    changes=np.r_[0,np.flatnonzero((codes[1:]!=codes[:-1])|(reason[1:]!=reason[:-1]))+1,n]
    timeline=[[int(a),int(b),int(codes[a]),int(reason[a])] for a,b in zip(changes[:-1],changes[1:])]
    if sum(b-a for a,b,_,_ in timeline)!=n or any(x[3] not in REASONS for x in timeline):
        raise AssertionError('Timeline partition mismatch')
    return stages,kept,timeline


def label_record(source,output,row):
    rel=data_relative(row)
    path=Path(source)/rel
    before=identity(path)
    table=pq.read_table(path,use_threads=False)
    cleaned,s1,counts,details=s1_repair(table,row)
    target=Path(output)/'labels_s1'/rel
    label_identity,label_sha=write_label(cleaned,target)
    if identity(path)!=before: raise RuntimeError(('Source label changed',path))
    record={'source_episode_index':int(row['episode_index']),'source_length':int(row['length']),
            'source_data_identity':before,'label_relative_path':rel,'repaired_label_identity':label_identity,
            'repaired_label_sha256':label_sha,'repair_counts':counts,'s1_removed_intervals':s1,
            'repair_details':details}
    return record,cleaned


def process_batch(job):
    source,output,rows,skip_s3,batch_path,reuse,resume=job
    batch_path=Path(batch_path)
    if resume and batch_path.is_file():
        with batch_path.open() as stream:
            cached=[json.loads(line) for line in stream]
        if len(cached)!=len(rows): raise ValueError('Checkpoint length mismatch')
        for rec,row in zip(cached,rows):
            if rec['source_episode_index']!=int(row['episode_index']): raise ValueError('Checkpoint indexing mismatch')
            if identity(Path(source)/rec['label_relative_path'])!=rec['source_data_identity']: raise ValueError('Source changed since checkpoint')
            if identity(Path(output)/'labels_s1'/rec['label_relative_path'])!=rec['repaired_label_identity']: raise ValueError('Output label changed since checkpoint')
        return cached
    records=[]
    for i,row in enumerate(rows):
        if reuse is None:
            rec,table=label_record(source,output,row)
        else:
            rec=dict(reuse[i])
            if rec['source_episode_index']!=int(row['episode_index']): raise ValueError('S1 manifest indexing mismatch')
            if identity(Path(source)/rec['label_relative_path'])!=rec['source_data_identity']: raise ValueError('Source changed since S1 export')
            target=Path(output)/'labels_s1'/rec['label_relative_path']
            if identity(target)!=rec['repaired_label_identity']: raise ValueError('Repaired label changed since S1 export')
            table=pq.read_table(target,columns=FILTER_FIELDS,use_threads=False)
        stage,kept,timeline=annotate(table,row,rec['s1_removed_intervals'],skip_s3)
        rec.update(stages=stage,accepted_intervals=kept,timeline=timeline,videos=video_records(source,row))
        records.append(rec)
    temp=batch_path.with_suffix('.jsonl.tmp')
    with temp.open('w') as stream:
        for rec in records: stream.write(json.dumps(rec,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    temp.replace(batch_path)
    return records
