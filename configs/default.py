"""
configs/default.py

Central configuration for the 3DSP-ViTPose graduation project.

This file contains all experiment constants, dataset paths, keypoint
definitions, training hyper-parameters, and visualization settings.

Environment variables can override the main paths:

    DATA_ROOT       Root directory for datasets.
    DSP_ROOT        Path to 3D-Shot-Posture-Dataset.
    CROWD_ROOT      Path to CrowdPose dataset.
    CHECKPOINT_DIR  Path where checkpoints are saved/loaded.
    ASSET_DIR       Path where figures and README assets are saved.
    LOG_DIR         Path where logs/history files are saved.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple


# =====================================================================
# Project metadata
# =====================================================================

PROJECT_NAME = "3DSP-ViTPose"
PROJECT_DESCRIPTION = (
    "Fine-tuned ViTPose-Base-Simple for 3D Shot Posture Estimation "
    "using 3DSP and CrowdPose."
)

SEED = 42


# =====================================================================
# Path helpers
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

_KAGGLE_WORKING = Path("/kaggle/working")
_IN_KAGGLE = _KAGGLE_WORKING.is_dir()


def _env_path(env_name: str, default: Path) -> str:
    """
    Return an absolute path from an environment variable if set,
    otherwise return the default path.
    """
    raw = os.getenv(env_name)
    path = Path(raw) if raw not in (None, "") else default
    return str(path.expanduser().absolute())


# ---------------------------------------------------------------------
# Runtime directories
# ---------------------------------------------------------------------
# On Kaggle, keep compatibility with the original notebook:
#   /kaggle/working/3D-Shot-Posture-Dataset
#   /kaggle/working/CrowdPose
#   /kaggle/working/vitpose_B_mix_best.pt
#
# Locally, default to:
#   ./data/3D-Shot-Posture-Dataset
#   ./data/CrowdPose
#   ./checkpoints
#   ./assets
#   ./logs
# ---------------------------------------------------------------------

DATA_ROOT = _env_path(
    "DATA_ROOT",
    _KAGGLE_WORKING if _IN_KAGGLE else REPO_ROOT / "data",
)

DSP_ROOT = _env_path(
    "DSP_ROOT",
    Path(DATA_ROOT) / "3D-Shot-Posture-Dataset",
)

CROWD_ROOT = _env_path(
    "CROWD_ROOT",
    Path(DATA_ROOT) / "CrowdPose",
)

CHECKPOINT_DIR = _env_path(
    "CHECKPOINT_DIR",
    _KAGGLE_WORKING if _IN_KAGGLE else REPO_ROOT / "checkpoints",
)

ASSET_DIR = _env_path(
    "ASSET_DIR",
    _KAGGLE_WORKING / "assets" if _IN_KAGGLE else REPO_ROOT / "assets",
)

LOG_DIR = _env_path(
    "LOG_DIR",
    _KAGGLE_WORKING / "logs" if _IN_KAGGLE else REPO_ROOT / "logs",
)


def ensure_runtime_dirs() -> None:
    """
    Create runtime directories if they do not exist.

    This is intentionally not called automatically on import.
    Call it from training/evaluation scripts when needed.
    """
    for directory in (DATA_ROOT, CHECKPOINT_DIR, ASSET_DIR, LOG_DIR):
        Path(directory).mkdir(parents=True, exist_ok=True)


# =====================================================================
# 3DSP dataset paths
# =====================================================================

DSP_TRAIN_DIR = os.path.join(DSP_ROOT, "3dsp", "train")

DSP_POSTURE_DIRNAME = "posture"
DSP_IMAGE_DIRNAME = "img"

DSP_LABEL_EXTENSION = ".json"
DSP_IMAGE_EXTENSION = ".jpg"


# =====================================================================
# CrowdPose dataset paths
# =====================================================================

CROWDPOSE_IMAGES_DIR = os.path.join(CROWD_ROOT, "images")
CROWDPOSE_JSON_PATH = os.path.join(
    CROWD_ROOT,
    "json",
    "crowdpose_train.json",
)


# =====================================================================
# Model configuration
# =====================================================================

MODEL_ID = "usyd-community/vitpose-base-simple"

NUM_KEYPOINTS = 17

INPUT_W = 192
INPUT_H = 256
INPUT_SIZE: Tuple[int, int] = (INPUT_W, INPUT_H)

# ImageNet normalization used by ViTPose.
MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# HuggingFace ViTPose forward pass uses dataset_index.
# In this project we always use the single fine-tuned 3DSP-style layout.
HF_DATASET_INDEX = 0


# =====================================================================
# Heatmap configuration
# =====================================================================

HEATMAP_SIGMA = 2.0
HEATMAP_TARGET_EPS = 1e-6
LOSS_EPS = 1e-6

# Temperature used in soft-argmax decoding.
SOFTARGMAX_TEMPERATURE = 100.0


# =====================================================================
# Dataset split configuration
# =====================================================================

TOTAL_SHOTS = 200
FRAMES_PER_SHOT = 20
TOTAL_FRAMES = TOTAL_SHOTS * FRAMES_PER_SHOT

# First 20 shuffled shots are held out for validation.
VAL_SHOTS = 20
TRAIN_SHOTS = TOTAL_SHOTS - VAL_SHOTS

# 3DSP is oversampled to balance it against CrowdPose.
DSP_OVERSAMPLE = 3

# Augmentation.
FLIP_PROB = 0.5


# =====================================================================
# CrowdPose filtering configuration
# =====================================================================

CROWDPOSE_KEYPOINT_COUNT = 14
CROWD_KEYPOINT_FLAT_LENGTH = CROWDPOSE_KEYPOINT_COUNT * 3  # 42

# Minimum number of visible CrowdPose keypoints required to keep a sample.
CROWD_MIN_KP = 8

# Minimum bounding-box size required to keep a sample.
CROWD_MIN_BBOX_W = 40
CROWD_MIN_BBOX_H = 60

# CrowdPose visibility flags used in the processed labels.
CROWD_HIDDEN_FLAG = 0
CROWD_VISIBLE_FLAG = 2

# Label visibility convention used in MixedPoseDataset.
LABEL_HIDDEN = 0.0
LABEL_VISIBLE = 2.0


# =====================================================================
# Training configuration
# =====================================================================

EPOCHS = 12
BATCH_SIZE = 8
NUM_WORKERS = 2

LR = 1e-5
WEIGHT_DECAY = 1e-4

WARMUP_EPOCHS = 1
MIN_LR_RATIO = 0.1

GRAD_CLIP = 1.0

DATALOADER_DROP_LAST = True

# Metric used to select the best checkpoint.
VALIDATION_METRIC = "val_px"
LOWER_IS_BETTER = True


# =====================================================================
# Checkpoint configuration
# =====================================================================

BEST_CHECKPOINT_NAME = "vitpose_B_mix_best.pt"
EPOCH_CHECKPOINT_TEMPLATE = "vitpose_B_mix_ep{epoch}.pt"

BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, BEST_CHECKPOINT_NAME)
HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "history.json")


# =====================================================================
# Keypoint definitions
# =====================================================================
# The 3DSP keypoint order below is inferred from the mapping logic in
# the notebook:
#
#   CrowdPose shoulders/elbows/wrists map to 3DSP indices 11-16.
#   CrowdPose hips/knees/ankles map to 3DSP indices 1-6.
#   CrowdPose head maps to 3DSP index 10.
#   CrowdPose neck maps to 3DSP index 9.
#
#   Indices 0, 7, and 8 are synthetic/derived joints:
#       0: pelvis / root, derived from hips
#       7: spine / torso center, derived from hips + shoulders
#       8: shoulder center, derived from shoulders
# =====================================================================

JOINT_NAMES: Tuple[str, ...] = (
    "pelvis",            # 0
    "left_hip",          # 1
    "left_knee",         # 2
    "left_ankle",        # 3
    "right_hip",         # 4
    "right_knee",        # 5
    "right_ankle",       # 6
    "spine",             # 7
    "shoulder_center",   # 8
    "neck",              # 9
    "head",              # 10
    "right_shoulder",    # 11
    "right_elbow",       # 12
    "right_wrist",       # 13
    "left_shoulder",     # 14
    "left_elbow",        # 15
    "left_wrist",        # 16
)

LEFT_JOINTS: Tuple[int, ...] = (1, 2, 3, 14, 15, 16)
RIGHT_JOINTS: Tuple[int, ...] = (4, 5, 6, 11, 12, 13)
MIDLINE_JOINTS: Tuple[int, ...] = (0, 7, 8, 9, 10)

# Joints that have exact direct correspondence for evaluation.
EXACT_JOINTS: Tuple[int, ...] = (
    1, 2, 3, 4, 5, 6,
    11, 12, 13, 14, 15, 16,
)

EVAL_JOINTS = EXACT_JOINTS

# These are excluded from the exact-joint metric because they are
# derived or not exactly matched across dataset formats.
NON_EXACT_JOINTS: Tuple[int, ...] = (0, 7, 8, 9, 10)

EXACT_JOINT_NAMES: Tuple[str, ...] = tuple(
    JOINT_NAMES[i] for i in EXACT_JOINTS
)


# =====================================================================
# CrowdPose -> 3DSP mapping
# =====================================================================
# CrowdPose 14-keypoint order:
#
#   0  left_shoulder
#   1  right_shoulder
#   2  left_elbow
#   3  right_elbow
#   4  left_wrist
#   5  right_wrist
#   6  left_hip
#   7  right_hip
#   8  left_knee
#   9  right_knee
#   10 left_ankle
#   11 right_ankle
#   12 head
#   13 neck
# =====================================================================

CROWDPOSE_JOINT_NAMES: Tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "head",
    "neck",
)

CROWD_TO_DSP: Dict[int, int] = {
    0: 14,   # left_shoulder   -> left_shoulder
    1: 11,   # right_shoulder  -> right_shoulder
    2: 15,   # left_elbow      -> left_elbow
    3: 12,   # right_elbow     -> right_elbow
    4: 16,   # left_wrist      -> left_wrist
    5: 13,   # right_wrist     -> right_wrist
    6: 1,    # left_hip        -> left_hip
    7: 4,    # right_hip       -> right_hip
    8: 2,    # left_knee       -> left_knee
    9: 5,    # right_knee      -> right_knee
    10: 3,   # left_ankle      -> left_ankle
    11: 6,   # right_ankle     -> right_ankle
    12: 10,  # head            -> head
    13: 9,   # neck            -> neck
}

# Joints that are synthetically derived after CrowdPose -> 3DSP mapping.
#
#   0 pelvis          <- mean of left/right hips
#   8 shoulder_center <- mean of left/right shoulders
#   7 spine           <- mean of hips + shoulders
#
CROWD_DERIVED_JOINTS: Dict[int, Tuple[int, ...]] = {
    0: (1, 4),
    8: (11, 14),
    7: (1, 4, 11, 14),
}


# =====================================================================
# COCO -> 3DSP mapping
# =====================================================================
# COCO 17-keypoint order:
#
#   0  nose
#   1  left_eye
#   2  right_eye
#   3  left_ear
#   4  right_ear
#   5  left_shoulder
#   6  right_shoulder
#   7  left_elbow
#   8  right_elbow
#   9  left_wrist
#   10 right_wrist
#   11 left_hip
#   12 right_hip
#   13 left_knee
#   14 right_knee
#   15 left_ankle
#   16 right_ankle
#
# The mapping below converts COCO keypoints to the 3DSP layout.
# Some 3DSP joints are produced by averaging multiple COCO joints.
# =====================================================================

COCO_TO_3DSP: Dict[int, Tuple[int, ...]] = {
    0: (11, 12),          # pelvis            <- left/right hip
    1: (11,),             # left_hip          <- COCO left_hip
    2: (13,),             # left_knee         <- COCO left_knee
    3: (15,),             # left_ankle        <- COCO left_ankle
    4: (12,),             # right_hip         <- COCO right_hip
    5: (14,),             # right_knee        <- COCO right_knee
    6: (16,),             # right_ankle       <- COCO right_ankle
    7: (5, 6, 11, 12),    # spine             <- shoulders + hips
    8: (5, 6),            # shoulder_center   <- shoulders
    9: (5, 6),            # neck              <- shoulders
    10: (0,),             # head              <- COCO nose
    11: (6,),             # right_shoulder    <- COCO right_shoulder
    12: (8,),             # right_elbow       <- COCO right_elbow
    13: (10,),            # right_wrist       <- COCO right_wrist
    14: (5,),             # left_shoulder     <- COCO left_shoulder
    15: (7,),             # left_elbow        <- COCO left_elbow
    16: (9,),             # left_wrist        <- COCO left_wrist
}


# =====================================================================
# Horizontal flip configuration
# =====================================================================
# FLIP_PAIRS_17 is an old-index -> new-index mapping.
#
# Usage in dataset augmentation:
#
#   for i in range(NUM_KEYPOINTS):
#       j = FLIP_PAIRS_17[i]
#       flipped_labels[j] = labels[i]
#       flipped_labels[j, 0] = 1.0 - labels[i, 0]
#
# This swaps left and right joints.
# =====================================================================

FLIP_PAIRS_17: Tuple[int, ...] = (
    0,   # pelvis            -> pelvis
    4,   # left_hip          -> right_hip
    5,   # left_knee         -> right_knee
    6,   # left_ankle        -> right_ankle
    1,   # right_hip         -> left_hip
    2,   # right_knee        -> left_knee
    3,   # right_ankle       -> left_ankle
    7,   # spine             -> spine
    8,   # shoulder_center   -> shoulder_center
    9,   # neck              -> neck
    10,  # head              -> head
    14,  # right_shoulder    -> left_shoulder
    15,  # right_elbow       -> left_elbow
    16,  # right_wrist       -> left_wrist
    11,  # left_shoulder     -> right_shoulder
    12,  # left_elbow        -> right_elbow
    13,  # left_wrist        -> right_wrist
)

FLIP_MAP: Dict[int, int] = {
    old: new
    for old, new in enumerate(FLIP_PAIRS_17)
}


# =====================================================================
# Skeleton visualization
# =====================================================================

SKELETON_EDGES: Tuple[Tuple[int, int], ...] = (
    # Left leg
    (0, 1),
    (1, 2),
    (2, 3),

    # Right leg
    (0, 4),
    (4, 5),
    (5, 6),

    # Spine / torso / head
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),

    # Right arm
    (8, 11),
    (11, 12),
    (12, 13),

    # Left arm
    (8, 14),
    (14, 15),
    (15, 16),
)


# =====================================================================
# Visualization defaults
# =====================================================================

KEYPOINT_RADIUS = 4
SKELETON_LINE_WIDTH = 2

QUALITATIVE_NUM_SAMPLES = 4
HEATMAP_NUM_SAMPLES = 2

# Default joints to display in heatmap visualizations.
#
#   0  pelvis
#   9  neck
#   10 head
#   16 left_wrist
#
HEATMAP_DISPLAY_JOINTS: Tuple[int, ...] = (0, 9, 10, 16)

HEATMAP_CMAP = "inferno"
PLOT_DPI = 150


# =====================================================================
# Basic sanity checks
# =====================================================================

assert len(JOINT_NAMES) == NUM_KEYPOINTS
assert len(FLIP_PAIRS_17) == NUM_KEYPOINTS
assert all(0 <= idx < NUM_KEYPOINTS for idx in FLIP_PAIRS_17)
assert all(
    0 <= a < NUM_KEYPOINTS and 0 <= b < NUM_KEYPOINTS
    for a, b in SKELETON_EDGES
)
assert len(EXACT_JOINTS) == 12
assert set(EXACT_JOINTS).issubset(set(range(NUM_KEYPOINTS)))