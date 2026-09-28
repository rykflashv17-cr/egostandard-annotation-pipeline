"""Read-only MINT-style EEF trajectory anomaly detection.

The block + Hampel rejection decisions follow the pinned MIT-licensed reference
in vendor/mint. This implementation adds reason masks and numerical validation;
it never interpolates, smooths, cuts, or overwrites a label.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation


REASONS = {
    1: "block_position_jump",
    2: "block_rotation_jump",
    4: "hampel_position_spike",
    8: "hampel_rotation_spike",
    16: "nonfinite_pose",
    32: "invalid_rotation_6d",
    64: "action_state_delay_mismatch",
}


@dataclass(frozen=True)
class Config:
    position_step_m: float = 0.15
    rotation_step_deg: float = 45.0
    max_block: int = 15
    position_recovery_m: float = 0.15
    hampel_k: float = 3.0
    hampel_position_floor_m: float = 0.10
    hampel_rotation_floor_deg: float = 30.0
    delay_atol: float = 1e-6

    def json(self):
        return asdict(self)


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def names(code):
    return [name for bit, name in REASONS.items() if int(code) & bit]


def rotations_6d(values):
    """EgoStandard stores the first two ROWS, not columns, of rotation matrices."""
    rows = np.asarray(values, dtype=np.float64).reshape(-1, 2, 3)
    valid = np.isfinite(rows).all(axis=(1, 2))
    clean = np.where(np.isfinite(rows), rows, 0.0)
    first = clean[:, 0]
    n1 = np.linalg.norm(first, axis=1)
    first = first / np.maximum(n1[:, None], 1e-12)
    second = clean[:, 1] - (first * clean[:, 1]).sum(axis=1)[:, None] * first
    n2 = np.linalg.norm(second, axis=1)
    second = second / np.maximum(n2[:, None], 1e-12)
    valid &= (n1 > 1e-8) & (n2 > 1e-8)
    matrices = np.stack([first, second, np.cross(first, second)], axis=1)
    # A malformed 6D representation is a data issue, not a motion jump.
    error = np.max(np.abs(matrices[:, :2] - clean), axis=(1, 2))
    valid &= error <= 1e-3
    matrices[~valid] = np.eye(3)
    return matrices, valid


def _steps(position, rotvec, indices, quaternion=False):
    gap = np.diff(indices).astype(np.float64)
    pos = np.linalg.norm(np.diff(position[indices], axis=0), axis=1) / gap
    rotation = Rotation.from_rotvec(rotvec[indices])
    if quaternion:
        quat = rotation.as_quat()
        dots = np.clip(np.abs(np.sum(quat[:-1] * quat[1:], axis=1)), 0.0, 1.0)
        rot = np.degrees(2 * np.arccos(dots)) / gap
    else:
        rot = np.degrees((rotation[:-1].inv() * rotation[1:]).magnitude()) / gap
    return pos, rot


def detect(position, rotvec, initial_valid=None, config=Config()):
    """Return per-frame uint8 reasons; normal frames have code 0.

    Thresholds are displacement/rotation PER FRAME, including division by frame
    gaps. They are not meters/second and are unrelated to the prior 0.2 s rule.
    The optimized block search preserves the upstream first-exit-jump behavior.
    """
    position = np.asarray(position, dtype=np.float64)
    rotvec = np.asarray(rotvec, dtype=np.float64)
    n = len(position)
    valid = np.ones(n, dtype=bool) if initial_valid is None else np.array(initial_valid, bool)
    valid &= np.isfinite(position).all(axis=1) & np.isfinite(rotvec).all(axis=1)
    reasons = np.zeros(n, dtype=np.uint8)
    indices = np.flatnonzero(valid)
    if len(indices) >= 4:
        pos, rot = _steps(position, rotvec, indices)
        jumps = np.flatnonzero((pos > config.position_step_m) | (rot > config.rotation_step_deg))
        i = 1
        while i < len(indices) - 1:
            incoming = np.searchsorted(jumps, i - 1)
            if incoming >= len(jumps):
                break
            i = int(jumps[incoming]) + 1
            if i >= len(indices) - 1:
                break
            outgoing = np.searchsorted(jumps, i)
            end = min(i + config.max_block, len(indices) - 1)
            if outgoing >= len(jumps) or jumps[outgoing] >= end:
                i += 1
                continue
            j = int(jumps[outgoing])
            recovery = np.linalg.norm(position[indices[j + 1]] - position[indices[i - 1]])
            if recovery < config.position_recovery_m:
                code = 0
                if pos[i - 1] > config.position_step_m or pos[j] > config.position_step_m:
                    code |= 1
                if rot[i - 1] > config.rotation_step_deg or rot[j] > config.rotation_step_deg:
                    code |= 2
                reasons[indices[i:j + 1]] |= code
                i = j + 1
            else:
                i += 1
        valid[reasons != 0] = False

    indices = np.flatnonzero(valid)
    thresholds = {"position_m_per_frame": None, "rotation_deg_per_frame": None}
    if len(indices) >= 5:
        pos, rot = _steps(position, rotvec, indices, quaternion=True)
        for speed, floor, bit, key in (
            (pos, config.hampel_position_floor_m, 4, "position_m_per_frame"),
            (rot, config.hampel_rotation_floor_deg, 8, "rotation_deg_per_frame"),
        ):
            median = np.median(speed)
            threshold = max(floor, float(median + config.hampel_k * 1.4826 * np.median(np.abs(speed - median))))
            thresholds[key] = threshold
            big = speed > threshold
            spike = np.zeros(len(indices), bool)
            spike[1:-1] = big[:-1] & big[1:]
            spike[0] = big[0] & ~big[1]
            spike[-1] = big[-1] & ~big[-2]
            reasons[indices[spike]] |= bit
    return reasons, thresholds


def detect_pose(position, rotation6d, config=Config()):
    matrices, rotation_valid = rotations_6d(rotation6d)
    finite = np.isfinite(position).all(axis=1) & np.isfinite(rotation6d).all(axis=1)
    valid = finite & rotation_valid
    rotvec = Rotation.from_matrix(matrices).as_rotvec()
    reasons, thresholds = detect(position, rotvec, valid, config)
    reasons[~finite] |= 16
    reasons[finite & ~rotation_valid] |= 32
    pos_step = np.zeros(len(position), np.float64)
    rot_step = np.zeros(len(position), np.float64)
    if len(position) > 1:
        pair_valid = valid[1:] & valid[:-1]
        pos_step[1:] = np.where(pair_valid, np.linalg.norm(np.diff(position, axis=0), axis=1), np.nan)
        r = Rotation.from_rotvec(rotvec)
        rot_step[1:] = np.where(pair_valid, np.degrees((r[:-1].inv() * r[1:]).magnitude()), np.nan)
    return reasons, thresholds, pos_step, rot_step


def detect_hand(state_pos, state_rot, action_pos, action_rot, config=Config()):
    """Detect one continuous state trajectory including a distinct terminal target.

    Existing cut ends repeat the final state; natural delay1/trunc ends may have
    an action target beyond the final video frame. Never invent an extra image.
    A mismatching action trajectory is inspected independently and annotated.
    """
    n = len(state_pos)
    equal = np.all(np.isclose(action_pos[:-1], state_pos[1:], atol=config.delay_atol, rtol=0, equal_nan=True), axis=1)
    equal &= np.all(np.isclose(action_rot[:-1], state_rot[1:], atol=config.delay_atol, rtol=0, equal_nan=True), axis=1)
    clamped = bool(np.allclose(action_pos[-1], state_pos[-1], atol=config.delay_atol, rtol=0, equal_nan=True)
                   and np.allclose(action_rot[-1], state_rot[-1], atol=config.delay_atol, rtol=0, equal_nan=True))
    if clamped:
        pos, rot = state_pos, state_rot
    else:
        pos = np.concatenate([state_pos, action_pos[-1:]])
        rot = np.concatenate([state_rot, action_rot[-1:]])
    code, thresholds, pos_step, rot_step = detect_pose(pos, rot, config)
    state_code = code[:n].copy()
    action_code = np.zeros(n, np.uint8)
    action_code[:-1] = state_code[1:]
    action_code[-1] = code[n - 1] if clamped else code[n]
    mismatch = ~equal
    if mismatch.any():
        own, _, _, _ = detect_pose(action_pos, action_rot, config)
        action_code |= own
        action_code[np.flatnonzero(mismatch)] |= 64
    return {
        "state": state_code, "action": action_code,
        "position_step": pos_step, "rotation_step": rot_step,
        "thresholds": thresholds,
        "terminal_action": "repeats_last_state" if clamped else "target_beyond_last_video_frame",
        "external_terminal_target_code": int(code[-1]) if not clamped else 0,
        "delay_mismatch_rows": int(mismatch.sum()),
    }


def intervals(code, fps, position_step=None, rotation_step=None):
    out = []
    for start, end in runs(code != 0):
        combined = int(np.bitwise_or.reduce(code[start:end]))
        item = {
            "start_frame": start, "end_frame_exclusive": end,
            "start_seconds": start / fps, "end_seconds": end / fps,
            "frames": end - start, "reason_code": combined, "reasons": names(combined),
        }
        for key, values in (("peak_position_step_m", position_step), ("peak_rotation_step_deg", rotation_step)):
            if values is not None:
                # Include both the entry and exit edge of an anomalous block.
                v = values[max(1, start):min(len(values), end + 1)]
                v = v[np.isfinite(v)]
                item[key] = float(v.max()) if len(v) else None
        out.append(item)
    return out
