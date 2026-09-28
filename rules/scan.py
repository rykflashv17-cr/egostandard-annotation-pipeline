"""Scan accepted EgoStandard labels; write annotations only, never mutate input."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from detector import Config, REASONS, detect_hand, intervals, runs

SIDES = ("left", "right")
CAMERAS = ("observation.images.camera_reference", "observation.images.camera_aux_0")
FIELDS = ["frame_index", "episode_index", "timestamp", "index"] + [
    f"{prefix}.eef.{side}.{field}"
    for prefix in ("observation.state", "action")
    for side in SIDES
    for field in ("position", "rotation_6d")
]
FIELDS += [f"{prefix}.gripper.{side}.closedness" for prefix in ("observation.state", "action") for side in SIDES]
LABEL_SCHEMA = pa.schema([
    ("episode_index", pa.int64()), ("frame_index", pa.int64()),
    ("timestamp", pa.float64()), ("dataset_index", pa.int64()),
    ("state_left_reason", pa.uint8()), ("state_right_reason", pa.uint8()),
    ("action_left_reason", pa.uint8()), ("action_right_reason", pa.uint8()),
])


def identity(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def json_write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def column(table, key, width=None):
    a = table[key].combine_chunks()
    if a.null_count:
        raise ValueError(f"Null column: {key}")
    if width is not None:
        if not np.all(np.diff(a.offsets.to_numpy()) == width):
            raise ValueError(f"Invalid vector width: {key}")
        a = a.flatten()
    values = a.to_numpy(zero_copy_only=False)
    return values if width is None else values.reshape(len(table), width)


def episode(root, row, config):
    ep = int(row["episode_index"])
    relative = f'data/chunk-{row["data/chunk_index"]:03d}/file-{row["data/file_index"]:03d}.parquet'
    path = Path(root) / relative
    before = identity(path)
    table = pq.read_table(path, columns=FIELDS, use_threads=False)
    n = len(table)
    if n != row["length"] or n < 1:
        raise ValueError("Length mismatch")
    if not np.array_equal(column(table, "frame_index"), np.arange(n)):
        raise ValueError("Frame index mismatch")
    if not np.all(column(table, "episode_index") == ep):
        raise ValueError("Episode index mismatch")
    if not np.array_equal(column(table, "index"), np.arange(row["dataset_from_index"], row["dataset_to_index"])):
        raise ValueError("Global index mismatch")
    if not np.allclose(column(table, "timestamp"), np.arange(n) / 30, atol=0.001, rtol=0):
        raise ValueError("Timestamp mismatch")
    result = {"episode_index": ep, "frames": n, "status": "ok", "data_path": relative,
              "data_identity": before, "videos": {}, "hands": {}}
    for cam in CAMERAS:
        prefix = "videos/" + cam + "/"
        vp = f'videos/{cam}/chunk-{row[prefix + "chunk_index"]:03d}/file-{row[prefix + "file_index"]:03d}.mp4'
        if row[prefix + "from_timestamp"] != 0 or abs(row[prefix + "to_timestamp"] - n / 30) > 0.001:
            raise ValueError("This scanner requires one independent video per episode")
        if not (Path(root) / vp).is_file():
            raise FileNotFoundError(vp)
        result["videos"][cam] = vp
    masks = {}
    for side in SIDES:
        sp = column(table, f"observation.state.eef.{side}.position", 3).astype(np.float64)
        sr = column(table, f"observation.state.eef.{side}.rotation_6d", 6).astype(np.float64)
        ap = column(table, f"action.eef.{side}.position", 3).astype(np.float64)
        ar = column(table, f"action.eef.{side}.rotation_6d", 6).astype(np.float64)
        detected = detect_hand(sp, sr, ap, ar, config)
        sg = column(table, f"observation.state.gripper.{side}.closedness", 1)
        ag = column(table, f"action.gripper.{side}.closedness", 1)
        gripper_equal = np.isclose(ag[:-1, 0], sg[1:, 0], atol=config.delay_atol, rtol=0, equal_nan=True)
        detected["action"][np.flatnonzero(~gripper_equal)] |= 64
        masks["state_" + side] = detected["state"]
        masks["action_" + side] = detected["action"]
        result["hands"][side] = {
            "state_intervals": intervals(detected["state"], 30, detected["position_step"], detected["rotation_step"]),
            "action_row_intervals": intervals(detected["action"], 30),
            "state_suspect_frames": int(np.count_nonzero(detected["state"])),
            "action_suspect_rows": int(np.count_nonzero(detected["action"])),
            "terminal_action": detected["terminal_action"],
            "external_terminal_target_code": detected["external_terminal_target_code"],
            "eef_delay_mismatch_rows": detected["delay_mismatch_rows"],
            "gripper_delay_mismatch_rows": int((~gripper_equal).sum()),
            "hampel_thresholds": detected["thresholds"],
            "state_reason_counts": {name: int(np.count_nonzero(detected["state"] & bit)) for bit, name in REASONS.items()},
            "action_reason_counts": {name: int(np.count_nonzero(detected["action"] & bit)) for bit, name in REASONS.items()},
        }
    state_union = masks["state_left"] | masks["state_right"]
    action_union = masks["action_left"] | masks["action_right"]
    result["state_suspect_frames"] = int(np.count_nonzero(state_union))
    result["action_suspect_rows"] = int(np.count_nonzero(action_union))
    result["video_intervals"] = intervals(state_union, 30)
    marked = np.flatnonzero(state_union | action_union)
    labels = {
        "episode_index": np.full(len(marked), ep, np.int64), "frame_index": marked.astype(np.int64),
        "timestamp": marked / 30, "dataset_index": column(table, "index")[marked],
        **{key + "_reason": value[marked] for key, value in masks.items()},
    }
    if identity(path) != before:
        raise RuntimeError("Source file changed during scanning")
    result["input_unchanged_during_scan"] = True
    return result, pa.Table.from_pydict(labels, schema=LABEL_SCHEMA)


def scan_batch(root, out, batch_id, rows, config):
    started = time.monotonic()
    out = Path(out)
    target = out / "batches" / f"batch-{batch_id:05d}.jsonl"
    sparse = out / "labels" / f"batch-{batch_id:05d}.parquet"
    summary_path = out / "batches" / f"batch-{batch_id:05d}.done.json"
    temporary = target.with_suffix(".jsonl.tmp")
    counts = Counter()
    tables = []
    with temporary.open("w") as f:
        for row in rows:
            try:
                r, labels = episode(root, row, config)
                if len(labels):
                    tables.append(labels)
                counts["successful_episodes"] += 1
                counts["successful_frames"] += r["frames"]
                counts["state_suspect_frames"] += r["state_suspect_frames"]
                counts["action_suspect_rows"] += r["action_suspect_rows"]
                counts["episodes_with_state_anomaly"] += bool(r["state_suspect_frames"])
                counts["episodes_with_any_label_anomaly"] += bool(len(labels))
                counts["video_intervals"] += len(r["video_intervals"])
                for side, hand in r["hands"].items():
                    counts["external_terminal_target_anomalies"] += bool(hand["external_terminal_target_code"])
                    counts["eef_delay_mismatch_rows"] += hand["eef_delay_mismatch_rows"]
                    counts["gripper_delay_mismatch_rows"] += hand["gripper_delay_mismatch_rows"]
                    for name, count in hand["state_reason_counts"].items():
                        counts["state_hand_frames__" + name] += count
            except Exception as exc:
                r = {"status": "error", "episode_index": int(row["episode_index"]),
                     "frames": int(row["length"]), "error": f"{type(exc).__name__}: {exc}"}
                counts["error_episodes"] += 1
            counts["processed_episodes"] += 1
            counts["processed_frames"] += int(row["length"])
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    joined = pa.concat_tables(tables) if tables else pa.Table.from_pylist([], schema=LABEL_SCHEMA)
    tmp_sparse = sparse.with_suffix(".parquet.tmp")
    pq.write_table(joined, tmp_sparse, compression="zstd")
    if pq.ParquetFile(tmp_sparse).metadata.num_rows != len(joined):
        raise RuntimeError("Annotation row count mismatch")
    tmp_sparse.replace(sparse)
    temporary.replace(target)
    summary = {"batch_id": batch_id, "counts": dict(counts), "label_rows": len(joined),
               "elapsed_seconds": time.monotonic() - started,
               "annotation_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
               "labels_sha256": hashlib.sha256(sparse.read_bytes()).hexdigest()}
    # The completion marker is written last. Interrupted batches are rerun.
    json_write(summary_path, summary)
    return summary


def load_rows(source):
    columns = ["episode_index", "length", "data/chunk_index", "data/file_index", "dataset_from_index", "dataset_to_index"]
    columns += ["videos/" + cam + "/" + key for cam in CAMERAS
                for key in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    files = sorted((source / "meta/episodes").glob("chunk-*/file-*.parquet"))
    table = pa.concat_tables([pq.read_table(p, columns=columns, use_threads=False) for p in files])
    table = table.take(np.argsort(table["episode_index"].to_numpy()))
    info = json.loads((source / "meta/info.json").read_text())
    if info["fps"] != 30 or info["state_action_delay"] != 1 or info["codebase_version"] != "v3.0":
        raise ValueError("Expected EgoStandard LeRobot v3, 30 FPS, delay1")
    if len(table) != info["total_episodes"] or not np.array_equal(table["episode_index"].to_numpy(), np.arange(len(table))):
        raise ValueError("Episode count or ordering mismatch")
    if int(table["length"].to_numpy().sum()) != info["total_frames"]:
        raise ValueError("Dataset frame count mismatch")
    for cam in CAMERAS:
        prefix = "videos/" + cam + "/"
        pairs = np.c_[table[prefix + "chunk_index"].to_numpy(), table[prefix + "file_index"].to_numpy()]
        if len(np.unique(pairs, axis=0)) != len(table):
            raise ValueError("Shared video files require a different annotation mapping")
    return info, table, files


def export_reports(source, output, expected, counts):
    report = output / "reports"
    report.mkdir(exist_ok=True)
    source_map_file = source / "meta/episode_source_map.jsonl"
    source_map = source_map_file.open()
    mapping = None
    def get_mapping(ep):
        nonlocal mapping
        while mapping is None or mapping["episode_index"] < ep:
            line = source_map.readline()
            if not line:
                raise ValueError("Incomplete episode source map")
            mapping = json.loads(line)
        if mapping["episode_index"] != ep:
            raise ValueError("Source map ordering mismatch")
        return mapping

    common = ["episode_index", "original_source_episode_index", "task", "ego_video_path", "label_data_path"]
    episode_fields = common + ["frames", "state_suspect_frames", "action_suspect_rows", "video_interval_count"]
    span_fields = common + ["hand", "start_frame", "end_frame_exclusive", "start_seconds", "end_seconds",
                            "original_start_seconds", "original_end_seconds", "reason_code", "reasons",
                            "peak_position_step_m", "peak_rotation_step_deg"]
    action_fields = common + ["hand", "start_action_row", "end_action_row_exclusive", "row_start_seconds", "row_end_seconds",
                              "reason_code", "reasons", "terminal_action", "includes_external_terminal_target"]
    sparse_count = 0
    with (report / "affected_episodes.csv").open("w", encoding="utf-8-sig", newline="") as ef, \
         (report / "suspect_video_intervals.csv").open("w", encoding="utf-8-sig", newline="") as sf, \
         (report / "suspect_action_rows.csv").open("w", encoding="utf-8-sig", newline="") as af, \
         (report / "affected_episodes.jsonl").open("w") as jf, \
         (report / "errors.jsonl").open("w") as err:
        ew = csv.DictWriter(ef, fieldnames=episode_fields); ew.writeheader()
        sw = csv.DictWriter(sf, fieldnames=span_fields); sw.writeheader()
        aw = csv.DictWriter(af, fieldnames=action_fields); aw.writeheader()
        seen = 0
        for path in sorted((output / "batches").glob("batch-*.jsonl")):
            batch_id = int(path.stem.split("-")[1])
            labels = output / "labels" / f"batch-{batch_id:05d}.parquet"
            sparse_count += pq.ParquetFile(labels).metadata.num_rows
            with path.open() as f:
                for line in f:
                    r = json.loads(line)
                    seen += 1
                    if r["status"] != "ok":
                        err.write(line)
                        continue
                    if not (r["state_suspect_frames"] or r["action_suspect_rows"]):
                        continue
                    m = get_mapping(r["episode_index"])
                    common_row = {
                        "episode_index": r["episode_index"],
                        "original_source_episode_index": m.get("filter_source_episode_index", m.get("source_episode_index", r["episode_index"])),
                        "task": m["task"],
                        "ego_video_path": str(source / r["videos"][CAMERAS[0]]),
                        "label_data_path": str(source / r["data_path"]),
                    }
                    r["source_mapping"] = m
                    r["annotation_parquet"] = str(labels)
                    jf.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
                    ew.writerow(dict(common_row, frames=r["frames"], state_suspect_frames=r["state_suspect_frames"],
                                     action_suspect_rows=r["action_suspect_rows"], video_interval_count=len(r["video_intervals"])))
                    for hand, h in r["hands"].items():
                        for span in h["state_intervals"]:
                            values = {k: span[k] for k in span_fields if k in span}
                            values["reasons"] = "|".join(span["reasons"])
                            sw.writerow(dict(common_row, **values, hand=hand,
                                             original_start_seconds=m.get("filter_source_start_seconds", 0) + span["start_seconds"],
                                             original_end_seconds=m.get("filter_source_start_seconds", 0) + span["end_seconds"]))
                        for span in h["action_row_intervals"]:
                            aw.writerow(dict(common_row, hand=hand, start_action_row=span["start_frame"],
                                             end_action_row_exclusive=span["end_frame_exclusive"],
                                             row_start_seconds=span["start_seconds"], row_end_seconds=span["end_seconds"],
                                             reason_code=span["reason_code"], reasons="|".join(span["reasons"]),
                                             terminal_action=h["terminal_action"],
                                             includes_external_terminal_target=bool(h["external_terminal_target_code"] and span["end_frame_exclusive"] == r["frames"])))
        if seen != expected:
            raise ValueError("Report count mismatch")
    source_map.close()
    summary = {"status": "complete" if counts.get("error_episodes", 0) == 0 else "complete_with_errors",
               "mode": "detect_and_annotate_only", "source": str(source), "counts": dict(counts),
               "sparse_annotation_rows": sparse_count,
               "suspect_video_seconds": counts.get("state_suspect_frames", 0) / 30,
               "state_frame_ratio": counts.get("state_suspect_frames", 0) / max(1, counts.get("successful_frames", 0)),
               "original_videos_and_labels_modified": False,
               "annotation_meaning": "Suspected numeric EEF trajectory anomaly; not a visually confirmed tracking error or bad action.",
               "frames_and_intervals": "[start_frame, end_frame_exclusive); video time is local to accepted episode. Action row t targets state t+1, except documented terminal policy.",
               "reason_bits": REASONS}
    json_write(report / "summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--sample", type=int, default=0, help="0 scans all; positive selects a reproducible random pilot")
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if sys.flags.optimize:
        p.error("Do not use Python -O: validation assertions must remain enabled")
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        p.error("Output must be separate from input dataset")
    if args.workers < 1 or args.sample < 0:
        p.error("Invalid worker/sample count")
    info, table, meta_files = load_rows(source)
    if args.sample:
        selected = sorted(np.random.default_rng(args.seed).choice(len(table), min(args.sample, len(table)), replace=False))
        table = table.take(selected)
    config = Config()
    fingerprint = {str(f.relative_to(source)): identity(f) for f in meta_files + [source / "meta/info.json", source / "meta/episode_source_map.jsonl"]}
    manifest = {
        "mode": "detect_and_annotate_only", "source": str(source), "fps": 30,
        "sample": args.sample, "seed": args.seed, "parameters": config.json(),
        "expected_episodes": len(table), "expected_frames": int(table["length"].to_numpy().sum()),
        "input_metadata_identities": fingerprint,
        "coordinate_frame": "fixed episode frame (reference camera at episode start)",
        "trajectory": "per-hand EEF state sequence plus distinct terminal action target; action rows inherit next-state annotations",
        "upstream": json.loads((Path(__file__).parent / "vendor/mint/SOURCE.json").read_text()),
        "code_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__), Path(__file__).with_name("detector.py"))},
        "reason_bits": {str(k): v for k, v in REASONS.items()},
    }
    if output.exists():
        if not args.resume:
            p.error("Output already exists; use --resume or a new path")
        if json.loads((output / "config.json").read_text()) != manifest:
            p.error("Resume configuration/source/code mismatch")
        if (output / "COMPLETE.json").exists():
            p.error("Audit is already complete; validate it or choose a new output")
    else:
        output.mkdir(parents=True)
        json_write(output / "config.json", manifest)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sub in ("batches", "labels"):
        (output / sub).mkdir(exist_ok=True)
    started = time.monotonic()
    jobs, counts = [], Counter()
    for batch_id, batch in enumerate(table.to_batches(max_chunksize=1000)):
        done = output / "batches" / f"batch-{batch_id:05d}.done.json"
        if done.exists():
            saved = json.loads(done.read_text())
            for rel, key in ((f"batches/batch-{batch_id:05d}.jsonl", "annotation_sha256"), (f"labels/batch-{batch_id:05d}.parquet", "labels_sha256")):
                if hashlib.sha256((output / rel).read_bytes()).hexdigest() != saved[key]:
                    raise RuntimeError("Completed annotation checksum mismatch")
            counts.update(saved["counts"])
        else:
            jobs.append((batch_id, batch.to_pylist()))
    print(json.dumps({"event": "start", "expected_episodes": len(table), "pending_batches": len(jobs), "workers": args.workers}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        futures = {pool.submit(scan_batch, str(source), str(output), i, rows, config): i for i, rows in jobs}
        for future in as_completed(futures):
            result = future.result()
            counts.update(result["counts"])
            progress = {"status": "running", "expected_episodes": len(table), "counts": dict(counts),
                        "elapsed_seconds_this_run": round(time.monotonic() - started, 2)}
            json_write(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    if counts["processed_episodes"] != len(table) or counts["processed_frames"] != manifest["expected_frames"]:
        raise RuntimeError("Full coverage mismatch")
    for rel, before in fingerprint.items():
        if identity(source / rel) != before:
            raise RuntimeError("Input metadata changed during scan")
    summary = export_reports(source, output, len(table), counts)
    summary["elapsed_seconds_this_run"] = time.monotonic() - started
    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    json_write(output / "progress.json", summary)
    if summary["status"] == "complete":
        json_write(output / "COMPLETE.json", summary)
    else:
        json_write(output / "FAILED.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
