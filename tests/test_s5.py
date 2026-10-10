"""S5 interval-contract tests; the real-video splice is also audited on source MP4s."""

import csv
from fractions import Fraction
import gzip
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_s5
import s5


class S5TimelineTests(unittest.TestCase):
    def make_annotation(self, root, timeline):
        with gzip.open(root/'episodes.csv.gz', 'wt', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                'source_episode_index', 'source_frames', 'accepted_frames', 'rejected_frames',
                'reference_video_path', 'aux_video_path', 'repaired_label_path'))
            writer.writeheader()
            writer.writerow({'source_episode_index': 7, 'source_frames': 12,
                             'accepted_frames': 8, 'rejected_frames': 4,
                             'reference_video_path': '/source/reference.mp4',
                             'aux_video_path': '/source/aux.mp4',
                             'repaired_label_path': '/labels/episode-7.parquet'})
        with gzip.open(root/'timeline.csv.gz', 'wt', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                'source_episode_index', 'start_frame', 'end_frame_exclusive', 'status'))
            writer.writeheader()
            for start, end, status in timeline:
                writer.writerow({'source_episode_index': 7, 'start_frame': start,
                                 'end_frame_exclusive': end, 'status': status})

    def test_middle_rejection_makes_two_clips_and_merges_adjacent_keep(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_annotation(root, [(0, 3, 'keep'), (3, 4, 'keep'),
                                        (4, 8, 'reject'), (8, 12, 'keep')])
            episode, intervals = next(run_s5.episode_plans(root))
            self.assertEqual(episode['source_episode_index'], '7')
            self.assertEqual(intervals, [[0, 4], [8, 12]])
            summary = run_s5.summarize_plan(root)
            self.assertEqual(summary['new_video_files'], 4)
            self.assertEqual(summary['kept_frames'], 8)

    def test_gap_is_rejected_before_video_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_annotation(root, [(0, 4, 'keep'), (5, 8, 'reject'),
                                        (8, 12, 'keep')])
            with self.assertRaisesRegex(ValueError, 'gap/overlap'):
                list(run_s5.episode_plans(root))

    def test_packet_timestamp_maps_to_exact_frame(self):
        self.assertEqual(s5.packet_frame_index(5120, Fraction(1, 15360)), 10)
        with self.assertRaisesRegex(ValueError, 'Non-30-FPS'):
            s5.packet_frame_index(5130, Fraction(1, 15360))


if __name__ == '__main__':
    unittest.main()
