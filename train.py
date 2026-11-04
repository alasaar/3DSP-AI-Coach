"""
Main training script.

Usage:
    python train.py
"""

import glob
import math
import os
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.default import *
from data.crowdpose_preprocess import build_crowdpose_samples
from data.dataset import MixedPoseDataset
from models.vitpose import build_model
from utils.heatmap import make_targets, soft_argmax


def build_dataloaders():
    # ---- 3DSP split ----
    all_shots = sorted(glob.glob(os.path.join(DSP_ROOT, "3dsp", "train", "*")))
    random.seed(SEED)
    random.shuffle(all_shots)

    train_shots = all_shots[VAL_SHOTS:]
    val_shots   = all_shots[:VAL_SHOTS]

    dsp_train = [pp for s in train_shots
                 for pp in sorted(glob.glob(f"{s}/posture/*.json"))]
    dsp_val   = [pp for s in val_shots
                 for pp in sorted(glob.glob(f"{s}/posture/*.json"))]

    # oversample 3DSP to balance with CrowdPose
    dsp_train_over = dsp_train * DSP_OVERSAMPLE

    # ---- CrowdPose ----
    crowd_samples, _ = build_crowdpose_samples()

    # ---- DataLoaders ----
    train_ds = MixedPoseDataset(dsp_train_over, crowd_samples, augment=True)
    val_ds   = MixedPoseDataset(dsp_val, [])

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS)

    print(f"train: {len(train_ds)}  |  val: {len(val_ds)}  |  "
          f"steps/epoch: {len(train_dl)}")
    return train_dl, val_dl


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model, _ = build_model(device="cuda")
    train_dl, val_dl = build_dataloaders()

    total_steps  = EPOCHS * len(train_dl)
    warmup_steps = WARMUP_EPOCHS * len(train_dl)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(MIN_LR_RATIO, 0.5 * (1 + math.cos(math.pi * p)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_vpx = float("inf")
    history  = {"train_loss": [], "val_px": [], "lr": []}

    for epoch in range(EPOCHS):
        # ────────── train ──────────
        model.train()
        running = 0.0

        for batch in tqdm(train_dl, desc=f"E{epoch+1}/{EPOCHS}"):
            pv  = batch["pixel_values"].to("cuda")
            lbl = batch["labels"].to("cuda")
            ds  = torch.zeros(pv.size(0), dtype=torch.long, device="cuda")

            out = model(pixel_values=pv, dataset_index=ds)
            tgt, msk = make_targets(lbl, out.heatmaps.size(2),
                                    out.heatmaps.size(3))

            loss = (
                ((out.heatmaps - tgt) ** 2 * msk).sum()
                / (msk.sum() * out.heatmaps.size(2) * out.heatmaps.size(3)
                   + 1e-6)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            running += loss.item()

        train_loss = running / len(train_dl)

        # ────────── validate ──────────
        model.eval()
        vpx, vn = 0.0, 0

        with torch.no_grad():
            for batch in val_dl:
                pv  = batch["pixel_values"].to("cuda")
                lbl = batch["labels"].to("cuda")
                W   = batch["W"].float().to("cuda")
                H   = batch["H"].float().to("cuda")
                ds  = torch.zeros(pv.size(0), dtype=torch.long, device="cuda")

                hm     = model(pixel_values=pv, dataset_index=ds).heatmaps
                coords = soft_argmax(hm)

                dx = (coords[..., 0] - lbl[..., 0]) * W.unsqueeze(-1)
                dy = (coords[..., 1] - lbl[..., 1]) * H.unsqueeze(-1)

                B = pv.size(0)
                vpx += torch.sqrt(dx ** 2 + dy ** 2).mean().item() * B
                vn  += B

        val_px = vpx / vn
        cur_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_px"].append(val_px)
        history["lr"].append(cur_lr)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.5f}  "
              f"val_px={val_px:.2f}  lr={cur_lr:.2e}")

        # ---- checkpointing ----
        ckpt = os.path.join(CHECKPOINT_DIR, f"vitpose_B_mix_ep{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt)

        if val_px < best_vpx:
            best_vpx = val_px
            best_ckpt = os.path.join(CHECKPOINT_DIR, "vitpose_B_mix_best.pt")
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✓ new best → {best_ckpt}")

    print(f"\nBest val_px: {best_vpx:.2f}")
    return history


if __name__ == "__main__":
    train()