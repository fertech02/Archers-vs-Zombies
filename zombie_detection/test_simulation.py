"""
Simulation test for ZombieCNN detection.

Generates synthetic frames with known zombie positions (green rectangles),
runs the trained model, and reports detection metrics + saves annotated images.

Usage:
    python -m zombie_detection.test_simulation
or:
    python zombie_detection/test_simulation.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from zombie_detection.cnn import ZombieCNN, MAX_ZOMBIES
from zombie_detection.preprocessing import preprocess_obs, decode_detections

# ── config ────────────────────────────────────────────────────────────────────
WEIGHTS_PATH    = os.path.join(os.path.dirname(__file__), "zombie_cnn.pth")
CONF_THRESHOLD  = 0.5
FRAME_W, FRAME_H = 320, 180      # synthetic frame resolution
N_FRAMES        = 8               # number of synthetic test frames
MAX_ZOMBIES_PER_FRAME = 3
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), "sim_results")
# ──────────────────────────────────────────────────────────────────────────────


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two [x, y, w, h] boxes."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter   = inter_w * inter_h
    union   = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def make_synthetic_frame(
    n_zombies: int,
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a (H, W, 3) uint8 frame with `n_zombies` green rectangles on a
    random dark background.

    Returns:
        frame : (H, W, 3) uint8
        boxes : (n_zombies, 4) float32  [x, y, w, h] in pixel coords
    """
    if rng is None:
        rng = np.random.default_rng()

    # dark noisy background (simulate a gloomy game scene)
    frame = rng.integers(10, 60, (frame_h, frame_w, 3), dtype=np.uint8)

    box_size_min, box_size_max = 20, 50
    boxes = []
    for _ in range(n_zombies):
        bw = int(rng.integers(box_size_min, box_size_max))
        bh = int(rng.integers(box_size_min, box_size_max))
        bx = int(rng.integers(0, frame_w - bw))
        by = int(rng.integers(0, frame_h - bh))

        # bright green rectangle — stands out from dark background
        frame[by:by + bh, bx:bx + bw, 0] = 30
        frame[by:by + bh, bx:bx + bw, 1] = 200
        frame[by:by + bh, bx:bx + bw, 2] = 50

        boxes.append([float(bx), float(by), float(bw), float(bh)])

    return frame, np.array(boxes, dtype=np.float32).reshape(-1, 4)


def match_detections(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[int, int, int]:
    """
    Greedy matching of predictions to ground-truth boxes.
    Returns (true_positives, false_positives, false_negatives).
    """
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            s = iou(pred, gt)
            if s > best_iou:
                best_iou, best_j = s, j
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_j)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def save_frame_figure(
    frame: np.ndarray,
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    frame_idx: int,
    output_dir: str,
):
    fig, ax = plt.subplots(1, figsize=(6, 4))
    ax.imshow(frame)

    for box in gt_boxes:
        rect = patches.Rectangle(
            (box[0], box[1]), box[2], box[3],
            linewidth=2, edgecolor="lime", facecolor="none", label="GT",
        )
        ax.add_patch(rect)

    for box in pred_boxes:
        rect = patches.Rectangle(
            (box[0], box[1]), box[2], box[3],
            linewidth=2, edgecolor="red", facecolor="none",
            linestyle="--", label="Pred",
        )
        ax.add_patch(rect)

    # deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    ax.set_title(f"Frame {frame_idx}  |  GT={len(gt_boxes)}  Pred={len(pred_boxes)}")
    ax.axis("off")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return path


def run_simulation(
    n_frames: int = N_FRAMES,
    conf_threshold: float = CONF_THRESHOLD,
    iou_threshold: float = 0.3,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load model ────────────────────────────────────────────────────────────
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[ERROR] Weights not found at {WEIGHTS_PATH}")
        print("  Run `python -m zombie_detection.train` first.")
        return

    model = ZombieCNN(input_shape=(3, 84, 84)).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    print(f"Loaded weights from {WEIGHTS_PATH}")
    print(f"Running on: {device}")
    print(f"Confidence threshold: {conf_threshold}  IoU threshold: {iou_threshold}")
    print("-" * 60)

    rng = np.random.default_rng(seed)

    total_tp = total_fp = total_fn = 0
    all_confidences = []

    for i in range(n_frames):
        n_zombies = int(rng.integers(0, MAX_ZOMBIES_PER_FRAME + 1))
        frame, gt_boxes = make_synthetic_frame(n_zombies, rng=rng)

        # ── inference ─────────────────────────────────────────────────────────
        tensor = preprocess_obs(frame).to(device)
        with torch.no_grad():
            preds = model(tensor)               # (1, MAX_ZOMBIES, 5)

        # collect all confidences for stats
        confs = preds[0, :, 0].cpu().numpy()
        all_confidences.extend(confs.tolist())

        pred_boxes = decode_detections(
            preds, conf_threshold=conf_threshold,
            orig_w=FRAME_W, orig_h=FRAME_H,
        )

        tp, fp, fn = match_detections(gt_boxes, pred_boxes, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        saved_path = save_frame_figure(frame, gt_boxes, pred_boxes, i, OUTPUT_DIR)
        print(
            f"  Frame {i:2d}: GT={n_zombies}  Pred={len(pred_boxes)}"
            f"  TP={tp} FP={fp} FN={fn}  → {saved_path}"
        )

    # ── aggregate metrics ─────────────────────────────────────────────────────
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    avg_conf  = float(np.mean(all_confidences))
    high_conf = float(np.mean(np.array(all_confidences) >= conf_threshold))

    print("-" * 60)
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1        : {f1:.3f}")
    print(f"Avg slot confidence    : {avg_conf:.3f}")
    print(f"Fraction of slots above threshold: {high_conf:.3f}")
    print(f"\nAnnotated frames saved to: {OUTPUT_DIR}/")

    # ── note on synthetic vs real frames ─────────────────────────────────────
    print()
    print("NOTE: These synthetic frames use colored rectangles, not real game")
    print("frames. A model trained on real data will likely score low here —")
    print("that is expected. For a realistic test, place real game screenshots")
    print(f"(320×180 RGB PNGs) in {OUTPUT_DIR}/real/ and re-run with --real.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames",   type=int,   default=N_FRAMES)
    parser.add_argument("--conf",       type=float, default=CONF_THRESHOLD)
    parser.add_argument("--iou",        type=float, default=0.3)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    run_simulation(
        n_frames=args.n_frames,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        seed=args.seed,
    )
