"""
Train ZombieCNN on the collected zombie_dataset.

Fixed-slot detection loss:
  - Slot k is always matched to GT zombie k (ordering fixed by the dataset).
  - BCE loss for confidence, smooth-L1 loss for bounding-box regression.
  - Positive slots weighted by LAMBDA_CONF_POS.
  - Data augmentation: random horizontal flip + color jitter.
  - Cosine LR schedule + gradient clipping.

Usage:
    python -m zombie_detection.train
or:
    python zombie_detection/train.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from collect_dataset import load_dataset
from zombie_detection.cnn import ZombieCNN
from zombie_detection.dataset import ZombieDataset

SAVE_PATH = os.path.join(os.path.dirname(__file__), "zombie_cnn.pth")

# Loss hyper-parameters
LAMBDA_BBOX     = 5.0   # weight for bbox regression vs confidence loss
LAMBDA_CONF_POS = 1.0   # neutral: equal weight for positive and negative slots


# ── Fixed-slot detection loss ─────────────────────────────────────────────────

def detection_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    preds  : (B, MAX_ZOMBIES, 5)
    targets: (B, MAX_ZOMBIES, 5)  slot k: conf=1 if real zombie, 0 if empty

    Slot k in preds is compared directly to slot k in targets.
    Confidence: weighted BCE.
    Bbox: smooth-L1 on positive slots only.
    """
    conf_pred = preds[:, :, 0]                  # (B, MAX_ZOMBIES)
    conf_gt   = targets[:, :, 0]
    bbox_pred = preds[:, :, 1:]                 # (B, MAX_ZOMBIES, 4)
    bbox_gt   = targets[:, :, 1:]

    pos_mask = conf_gt > 0.5                    # (B, MAX_ZOMBIES)

    # ── confidence loss (BCE) ─────────────────────────────────────────────────
    conf_pred_c = conf_pred.clamp(1e-6, 1.0 - 1e-6)
    bce_all = F.binary_cross_entropy(conf_pred_c, conf_gt, reduction="none")

    weight = torch.ones_like(bce_all)
    weight[pos_mask] = LAMBDA_CONF_POS
    conf_loss = (bce_all * weight).mean()

    # ── bbox loss (smooth-L1 on positive slots only) ──────────────────────────
    if pos_mask.any():
        bbox_loss = F.smooth_l1_loss(
            bbox_pred[pos_mask], bbox_gt[pos_mask], reduction="mean"
        ) * LAMBDA_BBOX
    else:
        bbox_loss = torch.tensor(0.0, device=preds.device)

    return conf_loss + bbox_loss


# ── augmentation helpers ──────────────────────────────────────────────────────

def augment_batch(
    frames: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Random horizontal flip + color jitter applied per-batch.
    frames  : (B, 3, H, W) float32 in [0, 1]
    targets : (B, MAX_ZOMBIES, 5)
    """
    B = frames.shape[0]

    # horizontal flip with 50% probability per sample
    flip_mask = torch.rand(B) < 0.5
    if flip_mask.any():
        frames[flip_mask] = torch.flip(frames[flip_mask], dims=[-1])
        tgt_flipped = targets[flip_mask].clone()
        real = tgt_flipped[:, :, 0] > 0.5
        tgt_flipped[:, :, 1] = torch.where(
            real,
            1.0 - tgt_flipped[:, :, 1] - tgt_flipped[:, :, 3],
            tgt_flipped[:, :, 1],
        )
        targets[flip_mask] = tgt_flipped

    # mild color jitter: random brightness + contrast
    brightness    = 1.0 + (torch.rand(B, 1, 1, 1, device=frames.device) - 0.5) * 0.3
    contrast_shift = (torch.rand(B, 1, 1, 1, device=frames.device) - 0.5) * 0.15
    frames = (frames * brightness + contrast_shift).clamp(0.0, 1.0)

    return frames, targets


# ── training loop ─────────────────────────────────────────────────────────────

def train(
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 5e-4,
    val_fraction: float = 0.1,
    save_path: str = SAVE_PATH,
    warmup_epochs: int = 3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    print("Loading dataset...")
    frames, labels = load_dataset()
    print(f"  {len(frames)} frames  |  batch_size={batch_size}  epochs={epochs}")

    dataset = ZombieDataset(frames, labels)
    n_val   = max(1, int(val_fraction * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model     = ZombieCNN(input_shape=(3, 90, 160)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # cosine LR with linear warmup
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        # training
        model.train()
        train_loss = 0.0
        n_valid = 0
        for frames_b, targets_b in train_loader:
            frames_b  = frames_b.to(device)
            targets_b = targets_b.to(device)

            frames_b, targets_b = augment_batch(frames_b, targets_b)

            preds = model(frames_b)
            loss  = detection_loss(preds, targets_b)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            n_valid += 1

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames_b, targets_b in val_loader:
                frames_b  = frames_b.to(device)
                targets_b = targets_b.to(device)
                preds     = model(frames_b)
                val_loss += detection_loss(preds, targets_b).item()

        train_loss /= max(1, n_valid)
        val_loss   /= len(val_loader)
        scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"]
        saved  = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            saved = "  [saved]"

        print(
            f"Epoch {epoch:3d}/{epochs}  lr={cur_lr:.2e}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}{saved}"
        )

    print(f"\nDone. Best val loss: {best_val:.4f}  |  Weights: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=5e-4)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
