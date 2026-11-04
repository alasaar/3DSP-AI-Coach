"""
Model factory – loads ViTPose-Base-Simple from HuggingFace,
re-initialises the prediction head, and optionally loads a
fine-tuned checkpoint.
"""

import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

from configs.default import MODEL_ID


def build_model(
    checkpoint: str = None,
    device: str = "cuda",
) -> tuple:
    """
    Returns
    -------
    model     : VitPoseForPoseEstimation
    processor : AutoProcessor  (needed only for baseline / HF pipeline)
    """
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model     = VitPoseForPoseEstimation.from_pretrained(
        MODEL_ID, device_map=device,
    )

    if checkpoint is None:
        # Re-initialise the 1×1 conv head for fine-tuning
        with torch.no_grad():
            torch.nn.init.normal_(model.head.conv.weight, mean=0.0, std=0.01)
            torch.nn.init.zeros_(model.head.conv.bias)
    else:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"[Model] Loaded checkpoint: {checkpoint}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] ViTPose-Base-Simple  |  {n_params:.1f} M parameters")
    return model, processor