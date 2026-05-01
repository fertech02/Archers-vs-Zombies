"""
Test ZombieCNN on REAL game frames from the KAZ environment.

Runs random agents through several steps, captures actual game screenshots,
runs the trained model, and saves annotated images comparing GT vs predictions.

Usage:
    python zombie_detection/real_game_test.py
    python zombie_detection/real_game_test.py --n_frames 20 --conf 0.4
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from pettingzoo.butterfly import knights_archers_zombies_v10

from zombie_detection.cnn import ZombieCNN
from zombie_detection.preprocessing import preprocess_obs, decode_detections

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "zombie_cnn.pth")
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "sim_results", "real")
FRAME_W, FRAME_H = 320, 180   # resize for display (same as training data)


def get_zombie_boxes(env, orig_hw, new_wh):
    """Extract and scale zombie bounding boxes from the game state."""
    game = env.unwrapped
    boxes = []
    for z in game.zombie_list:
        r = z.rect
        boxes.append([float(r.x), float(r.y), float(r.width), float(r.height)])
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.array(boxes, dtype=np.float32)
    sx = new_wh[0] / orig_hw[1]
    sy = new_wh[1] / orig_hw[0]
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    return boxes


def iou(box_a, box_b):
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter   = inter_w * inter_h
    union   = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return float(inter / union) if union > 0 else 0.0


def match_detections(gt_boxes, pred_boxes, iou_threshold=0.3):
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


def save_frame_figure(frame, gt_boxes, pred_boxes, frame_idx, output_dir, confidences=None):
    fig, ax = plt.subplots(1, figsize=(8, 5))
    ax.imshow(frame)

    for box in gt_boxes:
        rect = patches.Rectangle(
            (box[0], box[1]), box[2], box[3],
            linewidth=2, edgecolor="lime", facecolor="none", label="GT zombie",
        )
        ax.add_patch(rect)

    for k, box in enumerate(pred_boxes):
        conf_str = f" ({confidences[k]:.2f})" if confidences is not None else ""
        rect = patches.Rectangle(
            (box[0], box[1]), box[2], box[3],
            linewidth=2, edgecolor="red", facecolor="none",
            linestyle="--", label=f"Prediction{conf_str}",
        )
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 3, f"{confidences[k]:.2f}" if confidences is not None else "",
                color="red", fontsize=7, va="bottom")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
    ax.set_title(
        f"Real Game Frame {frame_idx}  |  GT={len(gt_boxes)} zombies  "
        f"Pred={len(pred_boxes)} detections",
        fontsize=10,
    )
    ax.axis("off")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"frame_{frame_idx:03d}.png")
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def run(n_frames=12, conf_threshold=0.5, iou_threshold=0.3, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(WEIGHTS_PATH):
        print(f"[ERROR] Weights not found at {WEIGHTS_PATH}")
        return

    model = ZombieCNN(input_shape=(3, 84, 84)).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {WEIGHTS_PATH}")
    print(f"Device: {device}  |  conf_threshold={conf_threshold}  iou_threshold={iou_threshold}")
    print("-" * 65)

    env = knights_archers_zombies_v10.env(
        render_mode="rgb_array",
        max_cycles=500,
        max_zombies=4,
        num_archers=2,
        num_knights=0,
    )

    rng = np.random.default_rng(seed)
    collected = 0
    total_tp = total_fp = total_fn = 0

    episode = 0
    while collected < n_frames:
        env.reset(seed=int(rng.integers(0, 2**31)))
        episode += 1
        step = 0

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            done = term or trunc
            env.step(None if done else env.action_space(agent).sample())

            # sample one frame per step, only on the first agent to avoid duplicates
            first_agent = env.agents[0] if env.agents else None
            if agent == first_agent and not done and step % 15 == 5:
                raw = env.render()                           # (H_orig, W_orig, 3)
                orig_hw = raw.shape[:2]                      # (H, W)

                frame_resized = np.array(
                    Image.fromarray(raw).resize((FRAME_W, FRAME_H), Image.BILINEAR),
                    dtype=np.uint8,
                )

                gt_boxes = get_zombie_boxes(env, orig_hw, (FRAME_W, FRAME_H))

                # skip frames with no zombies to focus on interesting cases
                if len(gt_boxes) == 0 and rng.random() < 0.7:
                    step += 1
                    continue

                tensor = preprocess_obs(frame_resized).to(device)
                with torch.no_grad():
                    preds = model(tensor)

                # get per-slot confidences for annotation
                confs_all = preds[0, :, 0].cpu().numpy()

                pred_boxes = decode_detections(
                    preds, conf_threshold=conf_threshold,
                    orig_w=FRAME_W, orig_h=FRAME_H,
                )

                # extract confidences for kept detections (sorted descending already)
                kept_confs = sorted(confs_all[confs_all >= conf_threshold].tolist(), reverse=True)

                tp, fp, fn = match_detections(gt_boxes, pred_boxes, iou_threshold)
                total_tp += tp
                total_fp += fp
                total_fn += fn

                path = save_frame_figure(
                    frame_resized, gt_boxes, pred_boxes,
                    collected, OUTPUT_DIR, confidences=kept_confs,
                )

                status = f"TP={tp} FP={fp} FN={fn}"
                print(
                    f"  Frame {collected:2d} (ep {episode} step {step:3d}): "
                    f"GT={len(gt_boxes)} Pred={len(pred_boxes)}  {status}  -> {os.path.basename(path)}"
                )

                collected += 1
                if collected >= n_frames:
                    break

            step += 1
            if done:
                break

    env.close()

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print("-" * 65)
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1        : {f1:.3f}")
    print(f"\nAnnotated screenshots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_frames",  type=int,   default=12)
    parser.add_argument("--conf",      type=float, default=0.5)
    parser.add_argument("--iou",       type=float, default=0.3)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    run(
        n_frames=args.n_frames,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        seed=args.seed,
    )
