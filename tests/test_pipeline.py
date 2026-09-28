"""Critical alignment, interval, switch, and checkpoint behavior tests."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import pipeline as p
from repair_core import list_array


def fixture(n=150,step=0.,angle=0.,invalid=None,external_bad=False,annotated=False):
    t=np.arange(n)
    ext=np.tile(np.eye(4),(n,1,1)).astype(np.float32)
    ext[:,0,3]=t*step
    if angle:ext[:,:3,:3]=Rotation.from_euler('y',(t*angle)[:,None],degrees=True).as_matrix()
    rot=np.tile(np.eye(3)[:2].reshape(6),(n,1)).astype(np.float32)
    data={'frame_index':pa.array(t),'episode_index':pa.array(np.zeros(n,np.int64)),
          'timestamp':pa.array((t/30).astype(np.float32)),'index':pa.array(t),
          p.EXT:pa.array(ext.tolist(),type=pa.list_(pa.field('element',pa.list_(pa.field('element',pa.float32()))))),
          'raw_hand_feature':list_array(np.tile(np.array([np.nan,1],np.float32),(n,1)))}
    for side,sign in (('left',-1),('right',1)):
        q=np.tile([sign*.1,0,2.],(n,1))
        pos=(np.einsum('nij,nj->ni',ext[:,:3,:3],q)+ext[:,:3,3]).astype(np.float32)
        if invalid and side=='left':pos[invalid[0]:invalid[1]]=np.nan
        action=np.r_[pos[1:],pos[-1:]]
        if external_bad and side=='left':action[-1]=np.nan
        for prefix,values in (('observation.state',pos),('action',action)):
            data[f'{prefix}.eef.{side}.position']=list_array(values)
            data[f'{prefix}.eef.{side}.rotation_6d']=list_array(rot)
            data[f'{prefix}.gripper.{side}.closedness']=list_array(np.zeros((n,1),np.float32))
    if annotated:
        for pre in ('observation.state','action'):data[f'{pre}.eef_valid']=list_array(np.ones((n,2),bool))
    intr={'matrix':[[50,0,50],[0,50,50],[0,0,1]],'dist_coeffs':[0]*5,
          'stored_image_size':{'width':100,'height':100}}
    row={'episode_index':0,'length':n,'data/chunk_index':0,'data/file_index':0,
         'dataset_from_index':0,'dataset_to_index':n,'camera_intrinsics':{'camera_reference':intr}}
    for view in p.VIEWS:
        prefix=f'videos/observation.images.{view}/'
        row.update({prefix+'chunk_index':0,prefix+'file_index':0,
                    prefix+'from_timestamp':1.25,prefix+'to_timestamp':1.25+n/30})
    return pa.table(data),row


class PipelineTests(unittest.TestCase):
    def assert_partition(self,n,timeline):
        cursor=0
        for a,b,code,reason in timeline:
            self.assertEqual(a,cursor);self.assertGreater(b,a);cursor=b
            self.assertIn(reason,p.REASONS)
        self.assertEqual(cursor,n)

    def test_interpolation_preserves_alignment_and_raw_nan(self):
        table,row=fixture(n=80,invalid=(10,12))
        out,bad,counts,detail=p.s1_repair(table,row)
        self.assertEqual(bad,[]);self.assertEqual(counts['repaired_hand_frames'],2)
        self.assertTrue(p.arrays_equal(table['raw_hand_feature'],out['raw_hand_feature']))
        for name in ('timestamp','index','frame_index','episode_index'):
            self.assertTrue(p.arrays_equal(table[name],out[name]))
        sp=p.column(out,'observation.state.eef.left.position',3)
        ap=p.column(out,'action.eef.left.position',3)
        self.assertTrue(np.array_equal(sp[1:],ap[:-1]))
        self.assertEqual(np.flatnonzero(p.column(out,'action.eef_repaired',2)[:,0]).tolist(),[9,10])
        self.assertFalse(p.column(out,'observation.state.hand_features_valid',2)[10,0])
        with tempfile.TemporaryDirectory() as directory:
            ident,digest=p.write_label(out,Path(directory)/'file.parquet')
            self.assertGreater(ident[2],0);self.assertEqual(len(digest),64)

    def test_unrepairable_rows_retained_and_terminal_action_rejected(self):
        table,row=fixture(n=80,invalid=(0,35))
        out,bad,counts,_=p.s1_repair(table,row)
        self.assertEqual(len(out),len(table));self.assertEqual(bad,[[0,35]])
        self.assertEqual(counts['unresolved_hand_frames'],35)
        self.assertTrue(np.isnan(p.column(out,'observation.state.eef.left.position',3)[:35]).all())
        stage,kept,timeline=p.annotate(out,row,bad)
        self.assertEqual(stage['S1']['removed_frames'],35);self.assert_partition(80,timeline)
        table,row=fixture(n=80,external_bad=True)
        out,bad,counts,_=p.s1_repair(table,row)
        self.assertEqual(bad,[[79,80]]);self.assertEqual(counts['external_terminal_rows_removed'],1)
        self.assertTrue(p.column(out,'observation.state.eef_valid',2)[-1].all())
        self.assertFalse(p.column(out,'action.eef_valid',2)[-1,0])

    def test_s3_switch_recomputes_s4_and_first_stage_claims(self):
        table,row=fixture(step=.015,annotated=True)
        stage,kept,timeline=p.annotate(table,row,[])
        self.assertEqual(stage['S3']['removed_frames'],150)
        self.assertEqual(stage['S4']['removed_frames'],0);self.assertEqual(kept,[])
        with patch.object(p,'detect_s3',side_effect=AssertionError('S3 was executed')):
            stage,kept,timeline=p.annotate(table,row,[],True)
        self.assertNotIn('S3',stage);self.assertEqual(stage['S4']['removed_frames'],150)
        self.assertEqual(timeline,[[0,150,4,41]])

    def test_rotation_branch_and_no_window_across_rejected_gap(self):
        table,row=fixture(angle=1.4,annotated=True)
        stage,kept,timeline=p.annotate(table,row,[])
        self.assertEqual(timeline,[[0,150,3,32]])
        stage,kept,timeline=p.annotate(table,row,[],True)
        self.assertEqual(timeline,[[0,150,4,42]])
        table,row=fixture(n=80,step=.008,annotated=True)
        stage,kept,timeline=p.annotate(table,row,[[40,41]],True)
        self.assertEqual(stage['S4']['removed_frames'],0)
        self.assertEqual(kept,[[0,40],[41,80]]);self.assert_partition(80,timeline)

    def test_checkpoint_reuses_labels_and_rejects_changed_files(self):
        table,row=fixture(n=80)
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);source=base/'source';output=base/'output'
            target=source/p.data_relative(row);target.parent.mkdir(parents=True)
            pq.write_table(table,target)
            checkpoint=base/'batch.jsonl'
            job=(str(source),str(output),[row],False,str(checkpoint),None,False)
            records=p.process_batch(job)
            with patch.object(p,'label_record',side_effect=AssertionError('Labels rewritten')):
                cached=p.process_batch((*job[:-1],True))
            self.assertEqual(records,cached)
            repaired=output/'labels_s1'/p.data_relative(row)
            before=p.identity(repaired)
            without=p.process_batch((str(source),str(output),[row],True,str(base/'off.jsonl'),records,False))
            self.assertEqual(before,p.identity(repaired));self.assertNotIn('S3',without[0]['stages'])
            with repaired.open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Output label changed'):
                p.process_batch((*job[:-1],True))


if __name__=='__main__':unittest.main()
