"""
biomechanics.py
Deterministic grounding layer: converts raw keypoints into biomechanical 
joint angles, phases, and rule-based technique checks.
"""
import numpy as np
from scipy.signal import savgol_filter

J = dict(pelvis=0, l_hip=1, l_knee=2, l_ankle=3, r_hip=4, r_knee=5, r_ankle=6,
         spine=7, sh_center=8, neck=9, head=10, r_sh=11, r_elbow=12, r_wr=13,
         l_sh=14, l_elbow=15, l_wr=16)

ANGLE_TRIPLETS = {
    "l_knee_flex":  (J["l_hip"],  J["l_knee"],  J["l_ankle"]),
    "r_knee_flex":  (J["r_hip"],  J["r_knee"],  J["r_ankle"]),
    "l_hip_flex":   (J["l_sh"],   J["l_hip"],   J["l_knee"]),
    "r_hip_flex":   (J["r_sh"],   J["r_hip"],   J["r_knee"]),
    "l_elbow_flex": (J["l_sh"],   J["l_elbow"], J["l_wr"]),
    "r_elbow_flex": (J["r_sh"],   J["r_elbow"], J["r_wr"]),
}

def smooth(traj, win=7, poly=2):
    if traj.shape[0] < win: return traj
    out = np.empty_like(traj)
    out[...,0] = savgol_filter(traj[...,0], win, poly, axis=0)
    out[...,1] = savgol_filter(traj[...,1], win, poly, axis=0)
    return out

def joint_angle(traj, a, b, c):
    v1 = traj[:,a] - traj[:,b]
    v2 = traj[:,c] - traj[:,b]
    cos = (v1*v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def trunk_lean(traj):
    v = traj[:, J["neck"]] - traj[:, J["pelvis"]]
    up = np.array([0.0, -1.0])
    cos = (v * up).sum(-1) / (np.linalg.norm(v, axis=-1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def generate_report(traj_s, conf):
    angles = {name: joint_angle(traj_s, *t) for name, t in ANGLE_TRIPLETS.items()}
    angles["trunk_lean"] = trunk_lean(traj_s)
    
    r_ank_speed = np.concatenate([[0], np.linalg.norm(np.diff(traj_s[:, J["r_ankle"]], axis=0), axis=1)])
    swing = r_ank_speed > 0.6 * r_ank_speed.max()
    s_start = int(np.argmax(swing))
    s_end = int(len(swing) - 1 - np.argmax(swing[::-1]))
    
    phases = {"approach": (0, s_start - 1), "strike": (s_start, s_end), "follow-through": (s_end + 1, len(traj_s) - 1)}
    
    CHECKS = [
        ("follow_ext", "Kicking leg collapses in follow-through", "follow-through", "r_knee_flex", "max", "<", 150),
        ("trunk_lean", "Trunk leans excessively during strike", "strike", "trunk_lean", "max", ">", 20),
    ]
    
    findings = []
    for cid, desc, ph, key, stat, op, thr in CHECKS:
        idx = np.arange(len(traj_s))
        if ph == "strike": idx = np.arange(s_start, s_end + 1)
        elif ph == "follow-through": idx = np.arange(s_end + 1, len(traj_s))
        
        if len(idx) == 0: continue
        ser = angles[key][idx]
        val = {"min": ser.min(), "max": ser.max(), "range": ser.max() - ser.min()}[stat]
        bad = (val > thr) if op == ">" else (val < thr)
        findings.append(dict(id=cid, description=desc, phase=ph, metric=key, stat=stat,
                             value=round(float(val), 1), threshold=thr, flagged=bool(bad)))
                             
    return {
        "n_frames": int(len(traj_s)),
        "mean_pose_confidence": round(float(conf.mean()), 3),
        "phases": {k: [int(v[0]), int(v[1])] for k, v in phases.items()},
        "angle_stats": {k: round(float(v.mean()), 1) for k, v in angles.items()},
        "findings": findings
    }