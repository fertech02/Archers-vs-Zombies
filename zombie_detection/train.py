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

SAVE_PATH       = os.path.join(os.path.dirname(__file__), "zombie_cnn.pth")
LAMBDA_BBOX     = 5.0    # bbox regression weight (positive cells only)
LAMBDA_OBJ      = 5.0    # confidence loss weight on cells with a zombie
LAMBDA_NOOBJ    = 0.5    # confidence loss weight on empty cells (most cells)


# ── YOLO-style detection loss ─────────────────────────────────────────────────

def _build_cell_targets(targets: torch.Tensor, gh: int, gw: int):
    """
    Map fixed-slot targets into a per-cell target tensor.

    Args:
        targets : (B, MAX_ZOMBIES, 5) [conf, x, y, w, h] in [0,1]; conf=1 if real zombie
        gh, gw  : grid dimensions

    Returns:
        cell_targets : (B, gh*gw, 5)  — populated only for cells containing a zombie
        cell_mask    : (B, gh*gw) bool — True where a zombie's top-left falls in that cell

    Note: if two zombies fall in the same cell the second is silently dropped (rare in KAZ).
    """
    B = targets.shape[0]
    device = targets.device
    N = gh * gw

    cell_targets = torch.zeros(B, N, 5, device=device)
    cell_mask    = torch.zeros(B, N, dtype=torch.bool, device=device)

    conf = targets[..., 0]
    x    = targets[..., 1].clamp(0.0, 1.0 - 1e-6)
    y    = targets[..., 2].clamp(0.0, 1.0 - 1e-6)
    w    = targets[..., 3]
    h    = targets[..., 4]

    valid = conf > 0.5
    if not valid.any():
        return cell_targets, cell_mask

    gx_idx = (x * gw).long().clamp(0, gw - 1)
    gy_idx = (y * gh).long().clamp(0, gh - 1)
    cell_idx = gy_idx * gw + gx_idx

    b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(cell_idx)

    flat_b    = b_idx[valid]
    flat_cell = cell_idx[valid]

    cell_targets[flat_b, flat_cell, 0] = 1.0
    cell_targets[flat_b, flat_cell, 1] = x[valid]
    cell_targets[flat_b, flat_cell, 2] = y[valid]
    cell_targets[flat_b, flat_cell, 3] = w[valid]
    cell_targets[flat_b, flat_cell, 4] = h[valid]
    cell_mask[flat_b, flat_cell] = True

    return cell_targets, cell_mask


def detection_loss(preds: torch.Tensor, targets: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    """
    preds  : (B, gh*gw, 5) — model output, sigmoid-bounded
    targets: (B, MAX_ZOMBIES, 5) — fixed-slot ground truth from ZombieDataset
    """
    cell_targets, cell_mask = _build_cell_targets(targets, gh, gw)

    conf_pred = preds[..., 0].clamp(1e-6, 1.0 - 1e-6)
    bce = F.binary_cross_entropy(conf_pred, cell_targets[..., 0], reduction="none")

    if cell_mask.any():
        obj_loss   = bce[cell_mask].mean()
        noobj_loss = bce[~cell_mask].mean()
    else:
        obj_loss   = torch.tensor(0.0, device=preds.device)
        noobj_loss = bce.mean()
    conf_loss = LAMBDA_OBJ * obj_loss + LAMBDA_NOOBJ * noobj_loss

    if cell_mask.any():
        bbox_loss = F.smooth_l1_loss(
            preds[..., 1:][cell_mask],
            cell_targets[..., 1:][cell_mask],
            reduction="mean",
        ) * LAMBDA_BBOX
    else:
        bbox_loss = torch.tensor(0.0, device=preds.device)

    return conf_loss + bbox_loss


# ── IoU-based validation metrics ──────────────────────────────────────────────

def _iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between sets of [x,y,w,h] boxes. a:(N,4) b:(M,4) → (N,M)."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros(a.shape[0], b.shape[0], device=a.device)
    a = a.unsqueeze(1)
    b = b.unsqueeze(0)
    ax1, ay1, aw, ah = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx1, by1, bw, bh = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    inter_w = (torch.min(ax1 + aw, bx1 + bw) - torch.max(ax1, bx1)).clamp(min=0)
    inter_h = (torch.min(ay1 + ah, by1 + bh) - torch.max(ay1, by1)).clamp(min=0)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def detection_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    conf_thr: float = 0.5,
    iou_thr: float = 0.5,
):
    """
    Greedy IoU matching per image.
    Returns (tp, fp, fn, sum_iou_of_matches, n_matches).
    """
    B = preds.shape[0]
    tp = fp = fn = 0
    sum_iou = 0.0
    n_match = 0

    for i in range(B):
        p = preds[i]
        t = targets[i]
        p_keep = p[:, 0] >= conf_thr
        t_keep = t[:, 0] > 0.5
        p_box = p[p_keep, 1:]
        t_box = t[t_keep, 1:]
        if p_keep.any():
            order = torch.argsort(-p[p_keep, 0])
            p_box = p_box[order]

        if len(t_box) == 0:
            fp += int(len(p_box)); continue
        if len(p_box) == 0:
            fn += int(len(t_box)); continue

        ious = _iou_matrix(p_box, t_box)               # (Np, Nt)
        matched = torch.zeros(t_box.shape[0], dtype=torch.bool, device=p.device)
        for j in range(p_box.shape[0]):
            avail = ious[j].clone()
            avail[matched] = -1.0
            best_iou, best_idx = avail.max(0)
            if best_iou.item() >= iou_thr:
                tp += 1; matched[best_idx] = True
                sum_iou += best_iou.item(); n_match += 1
            else:
                fp += 1
        fn += int((~matched).sum().item())

    return tp, fp, fn, sum_iou, n_match


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

    brightness     = 1.0 + (torch.rand(B, 1, 1, 1, device=frames.device) - 0.5) * 0.3
    contrast_shift = (torch.rand(B, 1, 1, 1, device=frames.device) - 0.5) * 0.15
    frames = (frames * brightness + contrast_shift).clamp(0.0, 1.0)

    return frames, targets


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
    gh, gw    = model.grid_h, model.grid_w
    print(f"Detection grid: {gh}×{gw} = {gh*gw} cells")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

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
            loss  = detection_loss(preds, targets_b, gh, gw)

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
        v_tp = v_fp = v_fn = 0
        v_iou_sum = 0.0; v_iou_n = 0
        with torch.no_grad():
            for frames_b, targets_b in val_loader:
                frames_b  = frames_b.to(device)
                targets_b = targets_b.to(device)
                preds     = model(frames_b)
                val_loss += detection_loss(preds, targets_b, gh, gw).item()

                tp, fp, fn, iou_s, iou_n = detection_metrics(preds, targets_b)
                v_tp += tp; v_fp += fp; v_fn += fn
                v_iou_sum += iou_s; v_iou_n += iou_n

        train_loss /= max(1, n_valid)
        val_loss   /= len(val_loader)
        scheduler.step()

        prec = v_tp / (v_tp + v_fp) if (v_tp + v_fp) > 0 else 0.0
        rec  = v_tp / (v_tp + v_fn) if (v_tp + v_fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        miou = v_iou_sum / v_iou_n if v_iou_n > 0 else 0.0

        cur_lr = optimizer.param_groups[0]["lr"]
        saved  = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            saved = "  [saved]"

        print(
            f"Epoch {epoch:3d}/{epochs}  lr={cur_lr:.2e}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"P={prec:.2f} R={rec:.2f} F1={f1:.2f} mIoU={miou:.2f}{saved}"
        )

    print(f"\nDone. Best val loss: {best_val:.4f}  |  Weights: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=5e-4)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)