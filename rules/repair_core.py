"""Repair only audited EEF outliers; preserve all unmarked numeric values.

Gap eligibility, linear position interpolation and shortest-path SO(3) SLERP
follow the pinned MINT fill_slerp_rot implementation. No global smoothing.
"""
import numpy as np
import pyarrow as pa
from scipy.spatial.transform import Rotation

from detector import rotations_6d, runs
from scan import column

SIDES = ("left", "right")
PREFIX = {"state": "observation.state", "action": "action"}


def fill_pose(position, rotation6d, codes, max_gap=30):
    position, rotation6d = np.asarray(position), np.asarray(rotation6d)
    codes = np.asarray(codes, np.uint8)
    if max_gap < 1 or len(position) != len(codes) or len(rotation6d) != len(codes):
        raise ValueError("Invalid interpolation arguments")
    if np.any(codes & 64):
        raise ValueError("Delay mismatch requires separate investigation")
    matrices, numerical_valid = rotations_6d(rotation6d)
    numerical_valid &= np.isfinite(position).all(axis=1)
    if np.any((codes == 0) & ~numerical_valid):
        raise ValueError("Unannotated invalid interpolation endpoint")
    pos, rot = position.copy(), rotation6d.copy()
    repaired = np.zeros(len(codes), bool)
    unresolved, filled_intervals = [], []
    indices, lower, upper = [], [], []
    for start, end in runs(codes != 0):
        reasons = []
        if start == 0:
            reasons.append("missing_left_endpoint")
        if end == len(codes):
            reasons.append("missing_right_endpoint")
        if end - start > max_gap:
            reasons.append("gap_exceeds_max_frames")
        item = {"start_frame": start, "end_frame_exclusive": end,
                "frames": end - start, "reason_code": int(np.bitwise_or.reduce(codes[start:end]))}
        if reasons:
            unresolved.append(dict(item, cannot_interpolate_reasons=reasons))
            continue
        filled_intervals.append(item)
        ids = np.arange(start, end)
        indices.append(ids)
        lower.append(np.full(len(ids), start - 1))
        upper.append(np.full(len(ids), end))
    if indices:
        ix, lo, hi = map(np.concatenate, (indices, lower, upper))
        alpha = ((ix - lo) / (hi - lo))[:, None]
        pos[ix] = (1 - alpha) * position[lo].astype(float) + alpha * position[hi].astype(float)
        left = Rotation.from_matrix(matrices[lo])
        right = Rotation.from_matrix(matrices[hi])
        delta = (left.inv() * right).as_rotvec()
        rot[ix] = (left * Rotation.from_rotvec(alpha * delta)).as_matrix()[:, :2].reshape(-1, 6)
        repaired[ix] = True
    valid = (codes == 0) | repaired
    if not np.isfinite(pos[valid]).all() or not rotations_6d(rot[valid])[1].all():
        raise ValueError("Interpolation produced an invalid pose")
    return pos, rot, valid, repaired, unresolved, filled_intervals


def list_array(values, arrow_type=None):
    values = np.asarray(values)
    typ = pa.from_numpy_dtype(values.dtype) if arrow_type is None else arrow_type.value_type
    target_type = arrow_type or pa.list_(pa.field("element", typ))
    return pa.ListArray.from_arrays(np.arange(0, values.size + 1, values.shape[1], dtype=np.int32),
                                    pa.array(values.reshape(-1), type=typ)).cast(target_type)


def repair_table(table, record, codes, max_gap=30):
    """Preserve raw hand features/gripper/cameras; masks explicitly concern EEF."""
    n = len(table)
    if n != record["frames"] or not np.array_equal(column(table, "frame_index"), np.arange(n)):
        raise ValueError("Episode frame mismatch")
    if not np.all(column(table, "episode_index") == record["episode_index"]):
        raise ValueError("Episode ID mismatch")
    result, details, flags = table, {}, {}
    for kind in PREFIX:
        for field in ("valid", "repaired", "anomaly_reason"):
            flags[f"{kind}_{field}"] = np.zeros((n, 2), np.uint8 if field == "anomaly_reason" else bool)
    for hand_index, side in enumerate(SIDES):
        hand = record["hands"][side]
        if hand["eef_delay_mismatch_rows"] or hand["gripper_delay_mismatch_rows"]:
            raise ValueError("Do not silently repair mismatching action/state labels")
        sp, sr, ap, ar = [column(table, f"{pre}.eef.{side}.{field}", width)
                          for pre in ("observation.state", "action")
                          for field, width in (("position", 3), ("rotation_6d", 6))]
        if not np.array_equal(sp[1:], ap[:-1], equal_nan=True) or not np.array_equal(sr[1:], ar[:-1], equal_nan=True):
            raise ValueError("Source action/state alignment is not exact")
        external = hand["terminal_action"] == "target_beyond_last_video_frame"
        if not external and (not np.array_equal(sp[-1], ap[-1], equal_nan=True) or not np.array_equal(sr[-1], ar[-1], equal_nan=True)):
            raise ValueError("Unexpected clamped terminal target")
        sc, ac = codes[f"state_{side}_reason"], codes[f"action_{side}_reason"]
        if not np.array_equal(sc[1:], ac[:-1]):
            raise ValueError("Annotation action/state mismatch")
        if ac[-1] != (hand["external_terminal_target_code"] if external else sc[-1]):
            raise ValueError("Terminal annotation mismatch")
        pos = np.r_[sp, ap[-1:]] if external else sp
        rot = np.r_[sr, ar[-1:]] if external else sr
        code = np.r_[sc, ac[-1:]] if external else sc
        cp, cr, valid, repaired, unresolved, filled = fill_pose(pos, rot, code, max_gap)
        action_indices = np.minimum(np.arange(n) + 1, len(pos) - 1)
        for kind, idx in (("state", np.arange(n)), ("action", action_indices)):
            for field, values in (("position", cp), ("rotation_6d", cr)):
                key = f"{PREFIX[kind]}.eef.{side}.{field}"
                old = column(table, key, values.shape[1])
                # Do not even round-trip unmarked values through a rotation conversion.
                new = old.copy()
                mask = repaired[idx]
                new[mask] = values[idx[mask]]
                result = result.set_column(result.schema.get_field_index(key), result.schema.field(key),
                                           list_array(new, result.schema.field(key).type))
            flags[f"{kind}_valid"][:, hand_index] = valid[idx]
            flags[f"{kind}_repaired"][:, hand_index] = repaired[idx]
            flags[f"{kind}_anomaly_reason"][:, hand_index] = code[idx]
        details[side] = {"terminal_action": hand["terminal_action"],
                         "filled_intervals": filled, "unresolved_intervals": unresolved,
                         "external_target_unresolved": bool(external and not valid[-1])}
    for key, values in flags.items():
        kind, field = key.split("_", 1)
        result = result.append_column(f"{PREFIX[kind]}.eef_{field}", list_array(values))
    # These are untouched raw keypoints, not keypoints reconstructed from repaired EEF.
    raw_hand_valid = flags["state_anomaly_reason"] == 0
    result = result.append_column("observation.state.hand_features_valid", list_array(raw_hand_valid))
    for side in SIDES:
        for field, width in (("position", 3), ("rotation_6d", 6)):
            s = column(result, f"observation.state.eef.{side}.{field}", width)
            a = column(result, f"action.eef.{side}.{field}", width)
            if not np.array_equal(s[1:], a[:-1], equal_nan=True):
                raise ValueError("Repaired action/state mismatch")
    return result, details, flags
