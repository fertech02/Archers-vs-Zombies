"""
Train ZombieCNN on the collected zombie_dataset.

Usage:
    python -m zombie_detection.train
or:
    python zombie_detection/train.py

Saves the best model weights to zombie_detection/zombie_cnn.pth.
These weights can then be loaded into the RLlib agent (rllib_model.py).
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from collect_dataset import load_dataset
from zombie_detection.cnn import ZombieCNN
from zombie_detection.dataset import ZombieDataset

SAVE_PATH = os.path.join(os.path.dirname(__file__), "zombie_cnn.pth")


def detection_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Combined confidence (BCE) + bounding-box (L1) loss.

    preds  : (B, MAX_ZOMBIES, 5)  — already sigmoid-ed by model.forward()
    targets: (B, MAX_ZOMBIES, 5)  — ground truth from ZombieDataset
    """
    conf_pred = preds[:, :, 0]            # (B, MAX_ZOMBIES)
    conf_gt   = targets[:, :, 0]

    bbox_pred = preds[:, :, 1:]           # (B, MAX_ZOMBIES, 4)
    bbox_gt   = targets[:, :, 1:]

    conf_loss = nn.functional.binary_cross_entropy(conf_pred, conf_gt)

    # bbox loss only on slots that contain a real zombie
    mask = conf_gt.unsqueeze(-1)          # (B, MAX_ZOMBIES, 1) — 1 for real, 0 for padding
    bbox_loss = (nn.functional.l1_loss(bbox_pred, bbox_gt, reduction="none") * mask).mean()

    return conf_loss + bbox_loss


def train(
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    save_path: str = SAVE_PATH,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    print("Loading dataset...")
    frames, labels = load_dataset()
    print(f"  {len(frames)} frames loaded")

    dataset = ZombieDataset(frames, labels)
    n_val   = max(1, int(val_fraction * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model     = ZombieCNN(input_shape=(3, 84, 84)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        # --- training ---
        model.train()
        train_loss = 0.0
        for frames_b, targets_b in train_loader:
            frames_b  = frames_b.to(device)
            targets_b = targets_b.to(device)
            preds = model(frames_b)
            loss  = detection_loss(preds, targets_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames_b, targets_b in val_loader:
                frames_b  = frames_b.to(device)
                targets_b = targets_b.to(device)
                preds     = model(frames_b)
                val_loss += detection_loss(preds, targets_b).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}", end="")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ saved → {save_path}", end="")
        print()

    print(f"\nTraining done. Best val loss: {best_val:.4f}")
    print(f"Weights saved to: {save_path}")


if __name__ == "__main__":
    train()
