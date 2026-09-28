"""Rule 3: stable hand-to-camera-forward cosines during sustained camera translation."""
from dataclasses import dataclass, asdict
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class Config:
    fps: int = 30
    window_seconds: float = 1.0
    translation_m: float = 0.40
    cosine_p95_p05: float = 0.03
    sustained_hit_seconds: float = 1.0
    hands: str = "both"
    rotation_deg: float | None = None
    translation_enabled: bool = True

    def validate(self):
        if self.fps != 30 or self.window_seconds < 1 or self.sustained_hit_seconds < 1:
            raise ValueError("30 FPS and windows/sustained hits >= 1 second required")
        if not (0 <= self.cosine_p95_p05 <= 2) or not np.isfinite(self.translation_m) or self.translation_m <= 0 or self.hands != "both":
            raise ValueError("Invalid thresholds; this approved rule requires both hands")
        if self.rotation_deg is not None and (not np.isfinite(self.rotation_deg) or not 0 < self.rotation_deg < 180):
            raise ValueError("Rotation threshold must be between 0 and 180 degrees")
        if not self.translation_enabled and self.rotation_deg is None:
            raise ValueError("At least one motion branch must be enabled")
        for x in (self.window_seconds, self.sustained_hit_seconds):
            if not np.isfinite(x) or abs(x*self.fps-round(x*self.fps)) > 1e-8:
                raise ValueError("Durations must be integral frame counts")

    def json(self):
        self.validate()
        gap=round(self.window_seconds*self.fps);minimum=round(self.sustained_hit_seconds*self.fps)
        return dict(asdict(self), window_frame_gap=round(self.window_seconds*self.fps),
                    min_consecutive_hit_windows=round(self.sustained_hit_seconds*self.fps),
                    cosine_definition="q=R_Episode_Camera.T@(p_hand_episode-camera_translation); cosine=q.z/norm(q)",
                    ego_direction="camera_reference optical forward +Z, not motion direction",
                    hand_validity=f"All {gap+1} frame positions per hand must be finite and eef_valid; zero-distance hand excluded",
                    motion_proxy="camera translation/rotation; no separately measured mobile base or body pose",
                    stability="Both hands separately: P95(cos)-P5(cos) <= threshold; numpy linear percentiles",
                    sustained_definition=f"Require >={minimum} consecutive matching window starts at 30 FPS; no gap bridging",
                    interval_convention="Include all images of qualifying windows; [first_start,last_start+window_gap+1); union overlaps/adjacency",
                    minimum_output_rejected_frames=round(self.window_seconds*self.fps)+round(self.sustained_hit_seconds*self.fps),
                    rotation_definition="Endpoint relative SO(3) rotation magnitude; not Euler subtraction or accumulated angle",
                    branch_combination="Each branch must independently sustain hits; qualified intervals are unioned",
                    other_gates="No radial/3D hand stability, gripper, object, visual, or activity conditions")


def runs(mask):
    d = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False]))
    return [(int(a), int(b)) for a,b in d.reshape(-1,2)]


def merge(spans):
    out = []
    for a,b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1],b)
        else:
            out.append([a,b])
    return out


def detect(ext, positions, valid, config=Config()):
    config.validate()
    ext, positions = np.asarray(ext,dtype=float), np.asarray(positions,dtype=float)
    valid = np.asarray(valid,dtype=bool)
    n = len(ext)
    if ext.shape != (n,4,4) or positions.shape != (n,2,3) or valid.shape != (n,2):
        raise ValueError("Unexpected input shape")
    if not np.isfinite(ext).all() or not np.allclose(ext[:,3],[0,0,0,1],atol=1e-5,rtol=0):
        raise ValueError("Invalid camera transforms")
    rotation, center = ext[:,:3,:3], ext[:,:3,3]
    if n and (np.max(np.abs(rotation.transpose(0,2,1)@rotation-np.eye(3))) > 1e-3 or np.any(np.linalg.det(rotation) <= 0)):
        raise ValueError("Camera rotation is not rigid")
    q = np.einsum("nji,nhj->nhi",rotation,positions-center[:,None,:])
    radius = np.linalg.norm(q,axis=2)
    good = valid & np.isfinite(q).all(axis=2) & (radius > 1e-6)
    cosine = np.zeros((n,2),dtype=float)
    np.divide(q[:,:,2],radius,out=cosine,where=good)
    cosine = np.clip(cosine,-1,1)
    gap, minimum = round(config.window_seconds*config.fps), round(config.sustained_hit_seconds*config.fps)
    if n <= gap:
        return dict(intervals=[],matching_window_runs=[],total_windows=0,raw_hit_windows=0,
                    qualified_hit_windows=0,invalid_windows=0,invalid_hand_frames=int((~good).sum()))
    windows = sliding_window_view(cosine,gap+1,axis=0)
    percentiles = np.percentile(windows,[5,95],axis=2,method="linear")
    spread = percentiles[1]-percentiles[0]
    window_valid = sliding_window_view(good,gap+1,axis=0).all(axis=(1,2))
    distance = np.linalg.norm(center[gap:]-center[:-gap],axis=1)
    stable = window_valid & (spread <= config.cosine_p95_p05).all(axis=1)
    branches=[]
    if config.translation_enabled:
        branches.append(("translation", distance > config.translation_m, distance, "max_camera_translation_m"))
    if config.rotation_deg is not None:
        relative=rotation[:-gap].transpose(0,2,1)@rotation[gap:]
        # atan2 is stable near 0/pi and avoids taking acos of a noisy float32 trace.
        skew=np.stack([relative[:,2,1]-relative[:,1,2],relative[:,0,2]-relative[:,2,0],relative[:,1,0]-relative[:,0,1]],axis=1)
        angles=np.rad2deg(np.arctan2(np.linalg.norm(skew,axis=1)/2,np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
        branches.append(("rotation", angles > config.rotation_deg + 1e-5, angles, "max_camera_rotation_deg"))
    raw=np.zeros(len(stable),bool);qualified_mask=raw.copy();reasons=[]
    for name,motion,metric,key in branches:
        hit=stable & motion;raw |= hit
        for a,b in runs(hit):
            if b-a < minimum:continue
            qualified_mask[a:b]=True
            reasons.append(dict(rule="stable_hand_cosine_camera_"+name,start=a,end=b+gap,
                                hit_start_frame=a,hit_end_frame_exclusive=b,consecutive_hit_windows=b-a,
                                **{key:float(metric[a:b].max())},
                                max_left_cosine_spread=float(spread[a:b,0].max()),
                                max_right_cosine_spread=float(spread[a:b,1].max())))
    return dict(intervals=merge([[r["start"],r["end"]] for r in reasons]),matching_window_runs=reasons,
                total_windows=len(stable),raw_hit_windows=int(raw.sum()),
                qualified_hit_windows=int(qualified_mask.sum()),invalid_windows=int((~window_valid).sum()),
                invalid_hand_frames=int((~good).sum()))
