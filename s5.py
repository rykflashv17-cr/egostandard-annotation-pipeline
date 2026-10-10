"""S5: export one independent video pair per retained timeline interval.

The source MP4s and S1 labels are immutable.  Boundary GOPs are re-encoded;
complete source GOPs are packet-copied.  Every published output is decoded and
checked against the source.  If a splice fails verification, the affected clip
is re-encoded in full rather than publishing an inaccurate cut.
"""

from bisect import bisect_left, bisect_right
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import av
import numpy as np


FPS = 30
VIEWS = {
    'camera_reference': ('reference_video_path', 'reference_from_timestamp'),
    'camera_aux_0': ('aux_video_path', 'aux_from_timestamp'),
}
VERSION = 's5-gop-splice-v1'


def file_identity(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as inp:
        for block in iter(lambda: inp.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def packet_frame_index(pts, time_base):
    exact = Fraction(pts) * time_base * FPS
    frame = round(exact)
    if abs(exact - frame) > Fraction(1, 1000):
        raise ValueError(f'Non-30-FPS packet timestamp: {exact}')
    return frame


def inspect_source(path, base, episode_frames):
    """Inspect source packet PTS and its independently decodable keyframes."""
    with av.open(str(path)) as container:
        streams = container.streams.video
        if len(streams) != 1 or len(container.streams) != 1:
            raise ValueError(f'Expected one video-only stream: {path}')
        stream = streams[0]
        if stream.codec_context.name != 'h264' or stream.average_rate != FPS:
            raise ValueError(f'Expected H.264 at 30 FPS: {path}')
        frames = set()
        keyframes = []
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            index = packet_frame_index(packet.pts, stream.time_base)
            if index in frames:
                raise ValueError(f'Duplicate frame PTS {index}: {path}')
            frames.add(index)
            if packet.is_keyframe:
                keyframes.append(index)
        expected = set(range(base, base + episode_frames))
        if not expected.issubset(frames):
            missing = sorted(expected - frames)
            raise ValueError(f'Source video lacks episode frames {missing[:8]}: {path}')
        if not keyframes or keyframes[0] > base:
            raise ValueError(f'No keyframe at or before episode start: {path}')
        return sorted(keyframes), {'codec': stream.codec_context.name,
                                   'width': stream.width, 'height': stream.height,
                                   'pix_fmt': stream.codec_context.pix_fmt,
                                   'time_base': str(stream.time_base),
                                   'keyframes': len(keyframes),
                                   'source_video_frames': len(frames)}


def ffmpeg_call(ffmpeg, args):
    cmd = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', *map(str, args)]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f'FFmpeg failed ({result.returncode}): {result.stderr[-4000:]}')


def encode_part(ffmpeg, source, seek_frame, start_frame, end_frame, target, crf, preset, mux='mpegts'):
    """Decode from a known keyframe and encode exactly [start_frame,end_frame)."""
    if not seek_frame <= start_frame < end_frame:
        raise ValueError('Invalid re-encode frame range')
    length = end_frame - start_frame
    seek_seconds = f'{seek_frame / FPS:.9f}'
    video_filter = (f'trim=start_frame={start_frame-seek_frame}:'
                    f'end_frame={end_frame-seek_frame},setpts=PTS-STARTPTS')
    args = ['-ss', seek_seconds, '-i', source, '-map', '0:v:0', '-an', '-vf', video_filter,
            '-frames:v', length, '-c:v', 'libx264', '-preset', preset, '-crf', crf,
            '-pix_fmt', 'yuv420p', '-g', '1', '-bf', '0', '-threads', '1',
            '-filter_threads', '1', '-f', mux]
    if mux == 'mp4':
        args += ['-movflags', '+faststart']
    ffmpeg_call(ffmpeg, args + [target])


def copy_gops(source, target, start_frame, end_frame):
    """Packet-copy [start,end) in presentation order, then mux in decode order."""
    with av.open(str(source)) as inp, av.open(str(target), 'w', format='mpegts') as out:
        stream = inp.streams.video[0]
        output_stream = out.add_stream_from_template(stream)
        seek_pts = round(Fraction(start_frame, FPS) / stream.time_base)
        inp.seek(seek_pts, stream=stream, backward=True, any_frame=False)
        copied = 0
        for packet in inp.demux(stream):
            if packet.pts is None or packet.dts is None:
                continue
            if packet_frame_index(packet.dts, stream.time_base) >= end_frame:
                break
            index = packet_frame_index(packet.pts, stream.time_base)
            if start_frame <= index < end_frame:
                packet.pts -= seek_pts
                packet.dts -= seek_pts
                packet.stream = output_stream
                out.mux(packet)
                copied += 1
        if copied != end_frame - start_frame:
            raise ValueError(f'Packet copy got {copied} frames, expected {end_frame-start_frame}: {source}')


def decode_frames(path, start=0, end=None):
    """Yield decoded frames whose timestamp maps to source frame [start,end)."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if start:
            seek_pts = round(Fraction(start, FPS) / stream.time_base)
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f'Decoded frame has no PTS: {path}')
            index = packet_frame_index(frame.pts, stream.time_base)
            if index < start:
                continue
            if end is not None and index >= end:
                break
            yield index, frame.to_ndarray(format='rgb24')


def verify_clip(source, output, start, end, copied_span):
    """Check every output frame against the corresponding source frame.

    Copied GOPs must be pixel-identical. Re-encoded boundary frames must have
    the same shape and at least 24 dB PSNR against their intended source frame.
    """
    expected = end - start
    source_iter = decode_frames(source, start, end)
    output_iter = decode_frames(output)
    min_psnr = float('inf')
    copied = encoded = 0
    previous_output_pts = None
    for offset in range(expected):
        try:
            source_index, original = next(source_iter)
            output_index, result = next(output_iter)
        except StopIteration as exc:
            raise ValueError(f'Truncated clip: {output}') from exc
        if source_index != start + offset or output_index != offset:
            raise ValueError(f'Frame PTS mismatch at output frame {offset}: {output}')
        if previous_output_pts is not None and output_index != previous_output_pts + 1:
            raise ValueError(f'Noncontiguous output frame PTS: {output}')
        previous_output_pts = output_index
        if original.shape != result.shape:
            raise ValueError(f'Output frame shape mismatch: {output}')
        in_copy = copied_span is not None and copied_span[0] <= source_index < copied_span[1]
        if in_copy:
            if not np.array_equal(original, result):
                raise ValueError(f'Copied GOP differs from original at {source_index}: {output}')
            copied += 1
        else:
            diff = original.astype(np.int16) - result.astype(np.int16)
            mse = float(np.mean(diff.astype(np.float32) ** 2))
            psnr = float('inf') if mse == 0 else 10 * np.log10(255 ** 2 / mse)
            if psnr < 24:
                raise ValueError(f'Re-encoded frame {source_index} PSNR {psnr:.2f} dB: {output}')
            min_psnr = min(min_psnr, psnr)
            encoded += 1
    if next(source_iter, None) is not None or next(output_iter, None) is not None:
        raise ValueError(f'Output/source has extra decoded frames: {output}')
    return {'frames': expected, 'copied_frames': copied, 'encoded_frames': encoded,
            'minimum_boundary_psnr_db': None if encoded == 0 else round(float(min_psnr), 2)}


def splice_view(ffmpeg, source, output, start, end, keyframes, work, crf, preset):
    """Build one MP4. Fallback to complete clip re-encode on a failed splice."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    before_start = keyframes[bisect_right(keyframes, start) - 1]
    next_start_pos = bisect_left(keyframes, start)
    next_start = keyframes[next_start_pos] if next_start_pos < len(keyframes) else end
    previous_end = keyframes[bisect_right(keyframes, end) - 1]
    middle_start = max(start, next_start)
    middle_end = min(end, previous_end)
    copied_span = None
    fallback_reason = None
    temp_output = work / 'output.mp4.part'
    try:
        if middle_start >= middle_end:
            raise ValueError('No complete GOP lies inside this clip')
        parts = []
        if start < middle_start:
            part = work / 'head.ts'
            encode_part(ffmpeg, source, before_start, start, middle_start, part, crf, preset)
            parts.append(part)
        part = work / 'middle.ts'
        copy_gops(source, part, middle_start, middle_end)
        parts.append(part)
        if middle_end < end:
            part = work / 'tail.ts'
            encode_part(ffmpeg, source, middle_end, middle_end, end, part, crf, preset)
            parts.append(part)
        concat_file = work / 'parts.txt'
        concat_file.write_text(''.join(f"file '{p.name}'\n" for p in parts))
        ffmpeg_call(ffmpeg, ['-f', 'concat', '-safe', '0', '-i', concat_file,
                             '-map', '0:v:0', '-an', '-c:v', 'copy', '-movflags', '+faststart',
                             '-f', 'mp4', temp_output])
        copied_span = (middle_start, middle_end)
        validation = verify_clip(source, temp_output, start, end, copied_span)
        method = 'gop_copy_boundary_encode' if validation['encoded_frames'] else 'gop_copy'
    except (ValueError, RuntimeError, av.error.FFmpegError) as exc:
        fallback_reason = f'{type(exc).__name__}: {exc}'[:500]
        temp_output.unlink(missing_ok=True)
        encode_part(ffmpeg, source, before_start, start, end, temp_output, crf, preset, mux='mp4')
        validation = verify_clip(source, temp_output, start, end, None)
        method = 'full_reencode_fallback'
    temp_output.replace(output)
    return dict(validation, method=method, fallback_reason=fallback_reason,
                output_bytes=output.stat().st_size, output_identity=file_identity(output))


def output_path(root, view, episode, segment):
    return (Path(root) / 'videos' / view / f'chunk-{episode//1000:03d}' /
            f'episode-{episode:06d}-segment-{segment:04d}.mp4')


def process_episode(task):
    """A worker owns all clips and both views for one source episode."""
    episode, intervals, root, ffmpeg, crf, preset, resume = task
    root = Path(root)
    ep = int(episode['source_episode_index'])
    frames = int(episode['source_frames'])
    started = time.perf_counter()
    checkpoint = root / 'checkpoints' / f'chunk-{ep//1000:03d}' / f'episode-{ep:06d}.json'
    source_identity = {view: file_identity(episode[fields[0]]) for view, fields in VIEWS.items()}
    source_identity['repaired_label'] = file_identity(episode['repaired_label_path'])
    if resume and checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved['source_identity'] != source_identity or saved['intervals'] != intervals:
            raise ValueError(f'S5 checkpoint/source/timeline differs for episode {ep}')
        for segment in saved['segments']:
            for item in segment['views'].values():
                if item['method'] != 'original_reference' and file_identity(item['path']) != item['output_identity']:
                    raise ValueError(f'S5 checkpoint video changed: {item["path"]}')
        return saved
    work = root / '.work' / f'episode-{ep:06d}'
    if work.exists():
        shutil.rmtree(work)
    segments = []
    metadata = {}
    keyframes = {}
    try:
        if intervals and intervals != [[0, frames]]:
            for view, (path_key, offset_key) in VIEWS.items():
                base_exact = Fraction(str(episode[offset_key])) * FPS
                base = round(base_exact)
                if abs(base_exact - base) > Fraction(1, 1000):
                    raise ValueError(f'Non-frame-aligned MP4 offset for episode {ep}: {view}')
                keyframes[view], metadata[view] = inspect_source(Path(episode[path_key]), base, frames)
        for sequence, (start, end) in enumerate(intervals):
            record = {'stage': 'S5', 'source_episode_index': ep, 'segment_index': sequence,
                      'source_start_frame': start, 'source_end_frame_exclusive': end,
                      'segment_frames': end-start, 'fps': FPS,
                      'repaired_label_path': episode['repaired_label_path'],
                      'repaired_label_source_rows': [start, end],
                      'trainable_delay1_rows_before_validity_flags': max(0, end-start-1),
                      'last_frame_action_inside_segment': False, 'views': {}}
            for view, (path_key, offset_key) in VIEWS.items():
                source = Path(episode[path_key])
                base = round(Fraction(str(episode[offset_key])) * FPS)
                if intervals == [[0, frames]]:
                    record['views'][view] = {'path': str(source), 'source_path': str(source),
                        'source_video_start_frame': base, 'source_video_end_frame_exclusive': base+frames,
                        'method': 'original_reference', 'frames': frames, 'copied_frames': frames,
                        'encoded_frames': 0, 'output_bytes': 0, 'output_identity': None}
                    continue
                target = output_path(root, view, ep, sequence)
                view_work = work / f'segment-{sequence:04d}' / view
                stats = splice_view(ffmpeg, source, target, base+start, base+end,
                                    keyframes[view], view_work, crf, preset)
                record['views'][view] = dict(stats, path=str(target), source_path=str(source),
                    source_video_start_frame=base+start,
                    source_video_end_frame_exclusive=base+end)
            segments.append(record)
        saved = {'status': 'complete', 'stage': 'S5', 'source_episode_index': ep,
                 'source_frames': frames, 'intervals': intervals,
                 'source_identity': source_identity, 'source_video_metadata': metadata,
                 'segments': segments, 'elapsed_seconds': time.perf_counter()-started}
        write_json(checkpoint, saved)
        return saved
    finally:
        if work.exists():
            shutil.rmtree(work)
