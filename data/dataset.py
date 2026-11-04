"""
MixedPoseDataset – unified PyTorch Dataset that serves both 3DSP
(small, domain-specific) and CrowdPose (large, in-the-wild) samples
with optional horizontal-flip augmentation.
"""

import json
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from configs.default import (
    INPUT_W, INPUT_H, MEAN, STD, FLIP_PAIRS_17, FLIP_PROB,
)

_MEAN = np.array(MEAN, dtype=np.float32)
_STD  = np.array(STD,  dtype=np.float32)


class MixedPoseDataset(Dataset):
    """
    Parameters
    ----------
    dsp_samples   : list[str]   – paths to 3DSP posture JSON files
    crowd_samples : list[tuple] – (img_path, bbox, kps_17×3)
    augment       : bool        – enable random horizontal flip
    """

    def __init__(
        self,
        dsp_samples: List[str],
        crowd_samples: List[Tuple],
        augment: bool = False,
    ):
        self.dsp     = dsp_samples
        self.crowd   = crowd_samples
        self.n_dsp   = len(dsp_samples)
        self.augment = augment

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_dsp + len(self.crowd)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        if idx < self.n_dsp:
            pv, lbl, W, H = self._load_3dsp(idx)
        else:
            pv, lbl, W, H = self._load_crowd(idx - self.n_dsp)

        # --- normalise to [0,1] then ImageNet stats ---
        arr = np.array(
            Image.fromarray(pv).resize((INPUT_W, INPUT_H), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        arr = (arr - _MEAN) / _STD
        pv  = torch.from_numpy(arr).permute(2, 0, 1).float()

        # --- augmentation: horizontal flip ---
        if self.augment and torch.rand(1).item() < FLIP_PROB:
            pv  = torch.flip(pv, dims=[-1])
            lbl = self._flip_labels(lbl)

        return {
            "pixel_values": pv,
            "labels":       lbl,
            "W":            float(W),
            "H":            float(H),
        }

    # ------------------------------------------------------------------
    def _load_3dsp(self, idx: int):
        pp  = self.dsp[idx]
        gt  = json.load(open(pp))
        ip  = pp.replace("/posture/", "/img/").replace(".json", ".jpg")
        img = Image.open(ip).convert("RGB")
        W, H = img.size

        rows = [
            [gt["keypoint_2d"][str(i)]["x"] / W,
             gt["keypoint_2d"][str(i)]["y"] / H,
             2.0]
            for i in range(17)
        ]
        lbl = torch.tensor(rows, dtype=torch.float32)
        return np.array(img), lbl, W, H

    # ------------------------------------------------------------------
    def _load_crowd(self, idx: int):
        ip, bbox, kps = self.crowd[idx]
        img = Image.open(ip).convert("RGB")
        x, y, w, h = bbox
        img = img.crop((int(x), int(y), int(x + w), int(y + h)))
        W, H = float(w), float(h)

        rows = []
        for i in range(17):
            px, py, v = kps[i]
            if v > 0:
                rows.append([(px - x) / W, (py - y) / H, 2.0])
            else:
                rows.append([0.0, 0.0, 0.0])
        lbl = torch.tensor(rows, dtype=torch.float32)
        return np.array(img), lbl, W, H

    # ------------------------------------------------------------------
    @staticmethod
    def _flip_labels(lbl: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(lbl)
        for i in range(17):
            j = FLIP_PAIRS_17[i]
            out[j]    = lbl[i].clone()
            out[j, 0] = 1.0 - lbl[i, 0]   # mirror x
        return out