"""
Keypoint remapping utilities (COCO ↔ 3DSP) used during evaluation.
"""

from typing import List, Tuple

from configs.default import COCO_TO_3DSP


def coco_to_3dsp(kps: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Convert 17 COCO keypoints → 17 3DSP-layout keypoints."""
    out = []
    for i in range(17):
        srcs = COCO_TO_3DSP[i]
        if len(srcs) == 1:
            out.append(kps[srcs[0]])
        else:
            cx = sum(kps[s][0] for s in srcs) / len(srcs)
            cy = sum(kps[s][1] for s in srcs) / len(srcs)
            out.append((cx, cy))
    return out