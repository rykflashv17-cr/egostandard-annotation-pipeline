#!/usr/bin/env python3
"""Export S5 video pairs from a completed S1-S4 EgoStandard timeline."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import fcntl
import gzip
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import imageio_ffmpeg

import s5


DEFAULT_ANNOTATIONS = Path(
    '/mnt/pfs/Data/ryk/egostandard/runs/labels_annotations_independent_B20_20260928/annotations/with_s3')
DEFAULT_OUTPUT = Path('/mnt/pfs/Data/ryk/egostandard/runs/video_segments_B20_s5')


def episode_plans(annotations):
    """Stream one original episode at a time, merging adjacent keep rows."""
    with gzip.open(annotations/'episodes.csv.gz', 'rt', newline='') as episodes_file, \
         gzip.open(annotations/'timeline.csv.gz', 'rt', newline='') as timeline_file:
        episode_rows = csv.DictReader(episodes_file)
        timeline_rows = csv.DictReader(timeline_file)
        if not {'source_episode_index', 'source_frames', 'reference_video_path',
                'aux_video_path', 'repaired_label_path'}.issubset(episode_rows.fieldnames or ()):
            raise ValueError('Missing S5 source episode fields')
        if not {'source_episode_index', 'start_frame', 'end_frame_exclusive',
                'status'}.issubset(timeline_rows.fieldnames or ()):
            raise ValueError('Missing S5 timeline fields')
        grouped = itertools.groupby(timeline_rows, lambda row: int(row['source_episode_index']))
        last_ep = -1
        for episode in episode_rows:
            ep = int(episode['source_episode_index'])
            frames = int(episode['source_frames'])
            if ep <= last_ep or frames < 1:
                raise ValueError(f'Invalid episode ordering/length: {ep}')
            last_ep = ep
            try:
                timeline_ep, rows = next(grouped)
            except StopIteration as exc:
                raise ValueError(f'Missing timeline for episode {ep}') from exc
            if timeline_ep != ep:
                raise ValueError(f'Episode/timeline ordering mismatch: {ep}, {timeline_ep}')
            cursor = 0
            intervals = []
            rejected = 0
            for row in rows:
                start, end = int(row['start_frame']), int(row['end_frame_exclusive'])
                if start != cursor or not start < end <= frames:
                    raise ValueError(f'Timeline gap/overlap in episode {ep} at {start}:{end}')
                cursor = end
                if row['status'] == 'keep':
                    if intervals and intervals[-1][1] == start:
                        intervals[-1][1] = end
                    else:
                        intervals.append([start, end])
                elif row['status'] == 'reject':
                    rejected += end-start
                else:
                    raise ValueError(f'Invalid timeline status in episode {ep}')
            if cursor != frames or sum(b-a for a,b in intervals)+rejected != frames:
                raise ValueError(f'Incomplete timeline for episode {ep}')
            if (int(episode['accepted_frames']) != frames-rejected or
                    int(episode['rejected_frames']) != rejected):
                raise ValueError(f'Per-episode time totals differ: {ep}')
            yield episode, intervals
        if next(grouped, None) is not None:
            raise ValueError('Extra timeline episodes')


def selected_plans(annotations, episode_ids, max_episodes):
    count = 0
    seen = set()
    for episode, intervals in episode_plans(annotations):
        ep = int(episode['source_episode_index'])
        if episode_ids is not None and ep not in episode_ids:
            continue
        seen.add(ep)
        yield episode, intervals
        count += 1
        if max_episodes and count >= max_episodes:
            break
    if episode_ids is not None and seen != episode_ids:
        raise ValueError(f'Episode IDs absent from annotation table: {sorted(episode_ids-seen)}')


def checked_annotation(annotations):
    marker = annotations/'COMPLETE.json'
    if not marker.is_file():
        raise ValueError(f'S1-S4 annotation is not complete: {marker}')
    complete = json.loads(marker.read_text())
    if complete.get('status') != 'complete':
        raise ValueError('Invalid S1-S4 completion marker')
    checks = complete.get('checksums', {})
    for name in ('timeline.csv.gz', 'episodes.csv.gz'):
        if name not in checks or s5.sha256(annotations/name) != checks[name]:
            raise ValueError(f'Annotation checksum mismatch: {name}')
    return complete


def summarize_plan(annotations, episode_ids=None, max_episodes=0):
    count = Counter()
    for episode, intervals in selected_plans(annotations, episode_ids, max_episodes):
        frames = int(episode['source_frames'])
        count['source_episodes'] += 1
        count['source_frames'] += frames
        count['kept_frames'] += sum(b-a for a,b in intervals)
        count['rejected_frames'] += frames-sum(b-a for a,b in intervals)
        count['segments'] += len(intervals)
        count['whole_source_references'] += intervals == [[0, frames]]
        count['new_video_files'] += 2*(len(intervals) if intervals != [[0, frames]] else 0)
    return dict(count)


def run(args):
    started = time.perf_counter()
    annotations = args.annotations.resolve()
    output = args.output.resolve()
    if output == annotations or output in annotations.parents or annotations in output.parents:
        raise ValueError('S5 output must be outside S1-S4 annotation directory')
    complete = checked_annotation(annotations)
    ids = set(args.episode) if args.episode else None
    if args.plan_only:
        result = summarize_plan(annotations, ids, args.max_episodes)
        print(json.dumps({'stage': 'S5', 'mode': 'plan_only', 'annotations': str(annotations),
                          **result}, ensure_ascii=False))
        return result
    ffmpeg = Path(args.ffmpeg or imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)
    version = subprocess.run([str(ffmpeg), '-version'], check=True, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.splitlines()[0]
    if not version.startswith('ffmpeg version'):
        raise ValueError(f'Unrecognized FFmpeg: {ffmpeg}')
    code_hashes = {name: s5.sha256(Path(__file__).with_name(name)) for name in ('s5.py', 'run_s5.py')}
    config = {'stage': 'S5', 'version': s5.VERSION, 'annotations': str(annotations),
              'annotation_timeline_sha256': complete['checksums']['timeline.csv.gz'],
              'annotation_episodes_sha256': complete['checksums']['episodes.csv.gz'],
              'source_stage_order': complete.get('stage_order'),
              'episode_ids': sorted(ids) if ids is not None else None,
              'max_episodes': args.max_episodes, 'fps': s5.FPS,
              'ffmpeg': str(ffmpeg), 'ffmpeg_version': version,
              'crf': args.crf, 'preset': args.preset,
              'whole_source_policy': 'reference original MP4; no duplicate video',
              'label_policy': 'reference S1 repaired label by source frame interval',
              'video_policy': 'copy complete GOPs; re-encode boundary GOPs; full re-encode fallback',
              'code_sha256': code_hashes}
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.s5.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = output/'config.json'
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError('S5 output fingerprint differs. Use a new output directory.')
            if not args.resume and not (output/'COMPLETE.json').exists():
                raise ValueError('Incomplete S5 output exists; use --resume')
        else:
            if any(path.name != '.s5.lock' for path in output.iterdir()):
                raise ValueError('Untracked S5 files exist in output directory')
            s5.write_json(config_path, config)
        if (output/'COMPLETE.json').exists():
            result = json.loads((output/'COMPLETE.json').read_text())
            print(json.dumps({'status': 'already_complete', 'summary': str(output/'summary.json')}))
            return result
        jobs = ((episode, intervals, str(output), str(ffmpeg), args.crf,
                 args.preset, args.resume)
                for episode, intervals in selected_plans(annotations, ids, args.max_episodes))
        totals = Counter()
        worker_seconds = 0.
        manifest_temp = output/'segments.jsonl.tmp'
        with manifest_temp.open('w') as manifest, \
             ProcessPoolExecutor(max_workers=args.workers) as pool:
            stream = iter(jobs)
            pending = [pool.submit(s5.process_episode, job)
                       for job in itertools.islice(stream, args.workers*2)]
            while pending:
                saved = pending.pop(0).result()
                ep = saved['source_episode_index']
                totals['source_episodes'] += 1
                totals['source_frames'] += saved['source_frames']
                totals['segments'] += len(saved['segments'])
                totals['kept_frames'] += sum(x['segment_frames'] for x in saved['segments'])
                worker_seconds += saved['elapsed_seconds']
                for segment in saved['segments']:
                    manifest.write(json.dumps(segment, ensure_ascii=False, allow_nan=False) + '\n')
                    for item in segment['views'].values():
                        if item['method'] == 'original_reference':
                            totals['original_references'] += 1
                        else:
                            totals['video_files_written'] += 1
                            totals['output_video_bytes'] += item['output_bytes']
                            totals['copied_frames_across_views'] += item['copied_frames']
                            totals['encoded_frames_across_views'] += item['encoded_frames']
                            totals[item['method']] += 1
                if totals['source_episodes'] % 100 == 0:
                    print(json.dumps({'stage': 'S5', 'status': 'running',
                        'episodes': totals['source_episodes'], 'segments': totals['segments'],
                        'output_video_bytes': totals['output_video_bytes']}), flush=True)
                try:
                    pending.append(pool.submit(s5.process_episode, next(stream)))
                except StopIteration:
                    pass
        manifest_temp.replace(output/'segments.jsonl')
        totals['rejected_frames'] = totals['source_frames']-totals['kept_frames']
        if ids is None and not args.max_episodes:
            if totals['source_episodes'] != complete['processed_episodes'] or \
               totals['kept_frames'] != complete['accepted_frames'] or \
               totals['rejected_frames'] != complete['total_removed_frames']:
                raise ValueError('S5 totals differ from completed S1-S4 summary')
        summary = {'status': 'complete', 'stage': 'S5', 'config': str(config_path),
                   'manifest': str(output/'segments.jsonl'), 'counts': dict(totals),
                   'kept_seconds': totals['kept_frames']/s5.FPS,
                   'rejected_seconds_from_existing_timeline': totals['rejected_frames']/s5.FPS,
                   'additional_rejected_seconds_by_s5': 0,
                   'wall_elapsed_seconds': time.perf_counter()-started,
                   'worker_cumulative_seconds': worker_seconds,
                   'manifest_sha256': s5.sha256(output/'segments.jsonl'),
                   'partial_export': ids is not None or bool(args.max_episodes)}
        s5.write_json(output/'summary.json', summary)
        s5.write_json(output/'COMPLETE.json', summary)
        print(json.dumps({'status': 'complete', 'stage': 'S5',
                          'summary': str(output/'summary.json'), 'counts': dict(totals)},
                         ensure_ascii=False), flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--ffmpeg', type=Path, help='FFmpeg executable; defaults to imageio-ffmpeg')
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--crf', type=int, default=18)
    parser.add_argument('--preset', default='veryfast')
    parser.add_argument('--episode', type=int, action='append', help='Export only this source episode; repeatable')
    parser.add_argument('--max-episodes', type=int, default=0, help='Export only first N selected episodes')
    parser.add_argument('--plan-only', action='store_true', help='Validate timeline and count output clips without writing video')
    parser.add_argument('--resume', action='store_true', help='Resume the same configuration from verified episode checkpoints')
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.crf <= 51 or args.max_episodes < 0 or \
       (args.episode and any(x < 0 for x in args.episode)):
        parser.error('Invalid workers, CRF, max-episodes, or episode ID')
    if sys.flags.optimize:
        parser.error('Do not use Python -O')
    run(args)


if __name__ == '__main__':
    main()
