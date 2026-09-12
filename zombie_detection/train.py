import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collect_dataset import load_dataset
from zombie_detection.cnn import ZombieCNN, ANCHOR_W, ANCHOR_H
from zombie_detection.dataset import ZombieDataset
from zombie_detection.utils import nms, DEFAULT_NMS_IOU

SAVE_PATH   = os.path.join(os.path.dirname(__file__), "zombie_cnn.pth")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "detector_config.json")

LAMBDA_XY       = 2.0   # Loss weight on the box centre offset (in cell units)
LAMBDA_OBJ      = 5.0   # Loss weight of objectness on cells with zombies
LAMBDA_NOOBJ    = 0.5   # Loss weight of objectness of cells without zombies

# Huber knee. This is the error above which the loss becomes L1 (constant gradient) instead of L2.
BBOX_BETA       = 0.1

# Operating point used for the per-epoch validation print and for model
# selection. The final sweep re-picks it on the validation split.
REF_CONF_THR    = 0.5
MATCH_IOU_THR   = 0.5

# Cells kept before NMS.
TOP_K           = 100

# Thresholds range values.
SWEEP_THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.05, 1.0, 0.05))


def build_cell_targets(targets: torch.Tensor, gh: int, gw: int):
    """
    ZombieDataset stores boxes as (objectness, x_left, y_top, w, h). YOLO makes
    the cell holding the box center responsible for it, so that the object
    sits in the middle of that cell's receptive field: convert first, then look
    up the cell.

    We write the centre coordinates in that cell of the tensor cell_targets and
    mark cell_mask[b, cell] = True to say we have a zombie.
    """
    B = targets.shape[0]
    device = targets.device
    N = gh * gw

    cell_targets = torch.zeros(B, N, 5, device=device)
    cell_mask    = torch.zeros(B, N, dtype=torch.bool, device=device)

    conf = targets[..., 0]
    w    = targets[..., 3]
    h    = targets[..., 4]

    # Top-left corner -> centre
    x = (targets[..., 1] + w / 2.0).clamp(0.0, 1.0 - 1e-6)
    y = (targets[..., 2] + h / 2.0).clamp(0.0, 1.0 - 1e-6)

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


def detection_loss(
    preds: torch.Tensor,
    conf_logits: torch.Tensor,
    targets: torch.Tensor,
    gh: int,
    gw: int,
) -> torch.Tensor:
    """
    preds       : (B, gh*gw, 5) - model output, (conf, x_centre, y_centre, w, h)
    conf_logits : (B, gh*gw)    - the same confidences before the sigmoid
    targets     : (B, MAX_ZOMBIES, 5) - fixed-slot ground truth from ZombieDataset

    Total loss: confidence (for each cell) + centre offset (only for cells with
    zombies). There is no size term: the sprite is a fixed 29x31 px box, so the
    model emits the constant anchor instead of regressing w and h.

    The offset term is not computed on the normalized image coordinates the
    model outputs but on the offsets in cell units. The model builds
    x = (gx + dx) / gw, so a loss on x sends only 1/gw of its gradient into the
    dx the head actually learns; multiplying by gw undoes that.
    """
    cell_targets, cell_mask = build_cell_targets(targets, gh, gw)

    bce = F.binary_cross_entropy_with_logits(
        conf_logits, cell_targets[..., 0], reduction="none"
    )

    if not cell_mask.any():
        # No zombie in the whole batch: only the "empty cell" term is defined
        return LAMBDA_NOOBJ * bce.mean()

    conf_loss = LAMBDA_OBJ * bce[cell_mask].mean() + LAMBDA_NOOBJ * bce[~cell_mask].mean()

    # Cell (row, col) of every positive, to undo the (gx + dx) / gw of the model.
    cell_idx = cell_mask.nonzero(as_tuple=False)[:, 1]
    gx_idx   = (cell_idx % gw).float()
    gy_idx   = (cell_idx // gw).float()

    p = preds[cell_mask]
    t = cell_targets[cell_mask]

    # Offsets in cell units. The target is in [0, 1) by construction of
    # build_cell_targets; the prediction can reach [-0.5, 1.5].
    pred_xy = torch.stack([p[:, 1] * gw - gx_idx, p[:, 2] * gh - gy_idx], dim=1)
    tgt_xy  = torch.stack([t[:, 1] * gw - gx_idx, t[:, 2] * gh - gy_idx], dim=1)

    xy_loss = F.smooth_l1_loss(pred_xy, tgt_xy, beta=BBOX_BETA, reduction="mean")

    return conf_loss + LAMBDA_XY * xy_loss


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    """IoU between sets of top-left [x,y,w,h] boxes. a:(N,4) b:(M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    a = a[:, None, :]
    b = b[None, :, :]

    ax1, ay1, aw, ah = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx1, by1, bw, bh = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    inter_w = np.clip(np.minimum(ax1 + aw, bx1 + bw) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay1 + ah, by1 + bh) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter

    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def decode_batch(preds_np: np.ndarray, conf_thr: float, nms_iou: float) -> list:

    out = []
    for i in range(preds_np.shape[0]):
        d = preds_np[i]
        d = d[d[:, 0] >= conf_thr]
        if len(d) == 0:
            out.append(np.zeros((0, 5), dtype=np.float32))
            continue

        if len(d) > TOP_K:
            d = d[np.argpartition(-d[:, 0], TOP_K)[:TOP_K]]

        boxes = np.stack([
            d[:, 1] - d[:, 3] / 2.0,
            d[:, 2] - d[:, 4] / 2.0,
            d[:, 3],
            d[:, 4],
        ], axis=1)

        keep = nms(boxes, d[:, 0], nms_iou)
        out.append(np.concatenate([d[keep, 0:1], boxes[keep]], axis=1).astype(np.float32))
    return out


def detection_metrics(decoded: list, targets_np: np.ndarray, iou_thr: float = MATCH_IOU_THR):
    """
    Greedy IoU matching per image, on boxes that already went through the
    deployment decode (threshold + NMS).
    Returns (tp, fp, fn, sum_iou_of_matches, n_matches).
    -> gt = ground truth (dataset real labels, top-left corner format)
    """
    tp = fp = fn = 0
    sum_iou = 0.0
    n_match = 0

    for i, p in enumerate(decoded):
        t = targets_np[i]
        t_box = t[t[:, 0] > 0.5, 1:]

        p_box = p[np.argsort(-p[:, 0]), 1:] if len(p) else p[:, 1:]

        if len(t_box) == 0:
            fp += len(p_box); continue   # no targets but all predictions

        if len(p_box) == 0:
            fn += len(t_box); continue   # targets but no predictions

        ious = iou_matrix(p_box, t_box)
        matched = np.zeros(len(t_box), dtype=bool)   # gt already assigned
        for j in range(len(p_box)):
            avail = ious[j].copy()
            avail[matched] = -1.0                    # exclude already assigned gt
            best_idx = int(avail.argmax())
            best_iou = float(avail[best_idx])
            if best_iou >= iou_thr:
                tp += 1; matched[best_idx] = True    # true positive
                sum_iou += best_iou; n_match += 1
            else:
                fp += 1                              # prediction without a nearby gt
        fn += int((~matched).sum())                  # never assigned gt

    return tp, fp, fn, sum_iou, n_match


def prf(tp: int, fp: int, fn: int, iou_sum: float, iou_n: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0,
        "MeanIou": iou_sum / iou_n if iou_n > 0 else 0.0,
    }


def split_by_chunk(
    chunk_ids: np.ndarray,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
):
    """
    Three-way split at the chunk (= group of episodes) level.

    Returns (train_idx, val_idx, test_idx) as frame index arrays.
    """
    uniq = np.unique(chunk_ids)
    if len(uniq) < 3:
        raise ValueError(
            f"Need at least 3 chunk files to build train/val/test, got {len(uniq)}. "
            "Re-collect with a smaller save_every."
        )

    perm = np.random.default_rng(seed).permutation(uniq)
    n_test = max(1, int(round(test_fraction * len(uniq))))
    n_val  = max(1, int(round(val_fraction  * len(uniq))))
    test_c, val_c, train_c = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    where = lambda cs: np.flatnonzero(np.isin(chunk_ids, cs))
    return where(train_c), where(val_c), where(test_c), (train_c, val_c, test_c)


@torch.no_grad()
def evaluate_split(
    model,
    loader,
    gh,
    gw,
    device,
    thresholds=(REF_CONF_THR,),
    nms_iou: float = DEFAULT_NMS_IOU,
):

    model.eval()
    total_loss = 0.0
    acc = {t: [0, 0, 0, 0.0, 0] for t in thresholds}

    for frames_b, targets_b in loader:
        frames_b  = frames_b.to(device)
        targets_b = targets_b.to(device)
        preds, conf_logits = model(frames_b, return_logits=True)
        total_loss += detection_loss(preds, conf_logits, targets_b, gh, gw).item()

        preds_np   = preds.detach().cpu().numpy()
        targets_np = targets_b.detach().cpu().numpy()

        for t in thresholds:
            stats = detection_metrics(decode_batch(preds_np, t, nms_iou), targets_np)
            for k, v in enumerate(stats):
                acc[t][k] += v

    return {
        "loss": total_loss / max(1, len(loader)),
        "by_threshold": {t: prf(*acc[t]) for t in thresholds},
    }


def pick_thresholds(by_threshold: dict) -> tuple:

    thrs = sorted(by_threshold)
    best_f1 = max(thrs, key=lambda t: by_threshold[t]["f1"])

    max_rec = max(by_threshold[t]["recall"] for t in thrs)
    knee = [t for t in thrs if by_threshold[t]["recall"] >= 0.99 * max_rec]
    best_rec = max(knee) if knee else best_f1

    return best_f1, best_rec


def augment_batch(
    frames: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    """
    Random horizontal flip + brightness/offset jitter applied per-batch.
    Horizontal flips adds geometric variety. Offset brightness breaks the
    correlation of episodes at the same distortion level having same photometric
    treatment.
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

    brightness = 1.0 + (torch.rand(B, 1, 1, 1, device=frames.device) - 0.5) * 0.3
    offset     = (torch.rand(B, 1, 1, 1, device=frames.device) - 0.5) * 0.15
    frames = (frames * brightness + offset).clamp(0.0, 1.0)

    return frames, targets


def train(
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 5e-4,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    save_path: str = SAVE_PATH,
    config_path: str = CONFIG_PATH,
    warmup_epochs: int = 3,
    split_seed: int = 42,
    nms_iou: float = DEFAULT_NMS_IOU,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    print("Loading dataset...")
    frames, labels, chunk_ids = load_dataset(return_chunks=True)
    print(f"  {len(frames)} frames  |  batch_size={batch_size}  epochs={epochs}")


    dataset = ZombieDataset(frames, labels)
    train_idx, val_idx, test_idx, (train_c, val_c, test_c) = split_by_chunk(
        chunk_ids, val_fraction, test_fraction, split_seed
    )
    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)
    test_ds  = Subset(dataset, test_idx)
    print(
        f"  split by chunk (seed {split_seed}): "
        f"train {len(train_idx)} frames / {len(train_c)} chunks | "
        f"val {len(val_idx)} / {len(val_c)} | test {len(test_idx)} / {len(test_c)}"
    )
    print(f"  test chunks held out: {sorted(test_c.tolist())}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    model     = ZombieCNN().to(device)
    gh, gw    = model.grid_h, model.grid_w
    print(f"Detection grid: {gh}x{gw} = {gh*gw} cells")
    print(f"Zombie size in cells: {ANCHOR_W * gw:.2f} x {ANCHOR_H * gh:.2f}")

    # AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs #linear warmup
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress)) # cosine aligning to gradually reduce it to 0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_f1 = -1.0
    for epoch in range(1, epochs + 1):

        # --    training    --
        model.train()
        train_loss = 0.0
        n_valid = 0
        for frames_b, targets_b in train_loader:
            frames_b  = frames_b.to(device)
            targets_b = targets_b.to(device)

            frames_b, targets_b = augment_batch(frames_b, targets_b)

            preds, conf_logits = model(frames_b, return_logits=True)
            loss  = detection_loss(preds, conf_logits, targets_b, gh, gw)

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
            optimizer.step()
            train_loss += loss.item()
            n_valid += 1

        # --    validation  --
        val = evaluate_split(model, val_loader, gh, gw, device, (REF_CONF_THR,), nms_iou)
        m   = val["by_threshold"][REF_CONF_THR]

        train_loss /= max(1, n_valid)
        scheduler.step()

        cur_lr = optimizer.param_groups[0]["lr"] # reads current lr
        saved  = ""
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save(model.state_dict(), save_path)
            saved = "  *"

        print(
            f"Epoch {epoch:3d}/{epochs}  lr={cur_lr:.2e}  "
            f"train={train_loss:.4f}  val={val['loss']:.4f}  "
            f"P={m['precision']:.2f} R={m['recall']:.2f} "
            f"F1={m['f1']:.2f} MeanIoU={m['MeanIou']:.2f}{saved}"
        )

    # -- threshold sweep on validation, with the selected weights --
    model.load_state_dict(torch.load(save_path, map_location=device))
    sweep = evaluate_split(model, val_loader, gh, gw, device, SWEEP_THRESHOLDS, nms_iou)
    print("\nValidation threshold sweep (decoded exactly as at inference):")
    print("  thr    P      R      F1     MeanIoU")
    for t in SWEEP_THRESHOLDS:
        s = sweep["by_threshold"][t]
        print(f"  {t:.2f}  {s['precision']:.3f}  {s['recall']:.3f}  {s['f1']:.3f}  {s['Meaniou']:.3f}")

    thr_f1, thr_rec = pick_thresholds(sweep["by_threshold"])
    config = {
        "input_size": [176, 320],
        "grid": [gh, gw],
        "nms_iou": nms_iou,
        "conf_threshold_f1": thr_f1,
        "conf_threshold_recall": thr_rec,
    }
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)
    print(
        f"\nOperating points -> {config_path}\n"
        f"  policy path (F1-optimal):        thr={thr_f1:.2f}  "
        f"F1={sweep['by_threshold'][thr_f1]['f1']:.3f}\n"
        f"  detector grading (recall knee):  thr={thr_rec:.2f}  "
        f"R={sweep['by_threshold'][thr_rec]['recall']:.3f}"
    )

    # -- test step: the best (by val F1) weights, on episodes never seen --
    test = evaluate_split(model, test_loader, gh, gw, device, (thr_f1, thr_rec), nms_iou)

    print(f"\nDone. Best val F1: {best_f1:.4f}  |  Weights: {save_path}")
    for name, t in (("F1 point", thr_f1), ("recall point", thr_rec)):
        s = test["by_threshold"][t]
        print(
            f"TEST ({len(test_ds)} frames, held-out episodes) @ {name} thr={t:.2f}  "
            f"loss={test['loss']:.4f}  P={s['precision']:.3f} R={s['recall']:.3f} "
            f"F1={s['f1']:.3f} MeanIoU={s['MeanIou']:.3f}"
        )
    return test


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--val_fraction",  type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--split_seed",    type=int,   default=42)
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
    )
