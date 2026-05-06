"""
Robust evaluation of ZombieCNN across all distortion levels.
Run: python test_detector.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from zombie_detection.cnn import ZombieCNN
from zombie_detection.preprocessing import preprocess_obs, decode_detections
from collect_dataset import load_dataset, get_zombie_boxes
from utils import create_environment

# ── config ────────────────────────────────────────────────────────────────────
CHECKPOINT   = "zombie_detection/zombie_cnn.pth"
CONF_THR     = 0.80
IOU_THR      = 0.4
MATCH_IOU    = 0.5    # IoU needed to count a detection as correct
N_EVAL_EPS   = 5      # episodes per distortion level
MAX_STEPS    = 200
ORIG_W, ORIG_H = 1280, 720


# ── load model ────────────────────────────────────────────────────────────────
def load_model(checkpoint):
    model = ZombieCNN(input_shape=(3, 90, 160))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    print(f"Loaded model from {checkpoint}")
    return model


# ── IoU between two single boxes [x,y,w,h] ───────────────────────────────────
def box_iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(ax2,  bx2);  iy2 = min(ay2,  by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


# ── match predictions to ground truth ─────────────────────────────────────────
def match_boxes(pred_boxes, gt_boxes, iou_threshold=MATCH_IOU):
    """
    Greedy matching: sort preds by confidence (already done upstream),
    match each pred to the best unmatched GT box.
    Returns (tp, fp, fn).
    """
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    tp = fp = 0
    for pb in pred_boxes:
        best_iou, best_idx = 0.0, -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = box_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_idx = iou, gi
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


# ── evaluate one distortion level ─────────────────────────────────────────────
def evaluate_level(model, distortion_level, n_episodes=N_EVAL_EPS):
    env = create_environment(
        max_cycles=MAX_STEPS,
        render_mode="rgb_array",
        distortion_level=distortion_level,
    )

    total_tp = total_fp = total_fn = 0
    frames_with_zombies = 0
    frames_all_found    = 0

    first_agent = env.possible_agents[0]

    for ep in range(n_episodes):
        env.reset(seed=ep + 100)   # different seeds from training data
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            done = term or trunc
            env.step(None if done else 0)

            if agent != first_agent:
                continue

            # ground truth boxes in original pixel space
            gt_raw = get_zombie_boxes(env)     # (N, 4) in 1280×720 space

            # model prediction
            raw_frame = env.render()           # (720, 1280, 3)
            inp       = preprocess_obs(raw_frame)
            with torch.no_grad():
                preds = model(inp)
            pred_boxes = decode_detections(
                preds,
                conf_threshold=CONF_THR,
                iou_threshold=IOU_THR,
                orig_w=ORIG_W,
                orig_h=ORIG_H,
            )                                  # (M, 4) in 1280×720 space

            tp, fp, fn = match_boxes(pred_boxes, gt_raw)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            if len(gt_raw) > 0:
                frames_with_zombies += 1
                if fn == 0:
                    frames_all_found += 1

    env.close()

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    pct_fully_detected = frames_all_found / frames_with_zombies if frames_with_zombies > 0 else 0.0

    return {
        "level":              distortion_level,
        "precision":          precision,
        "recall":             recall,
        "f1":                 f1,
        "tp":                 total_tp,
        "fp":                 total_fp,
        "fn":                 total_fn,
        "frames_all_found_%": pct_fully_detected * 100,
    }


# ── visualise a few detections ─────────────────────────────────────────────────
def visualise_detections(model, distortion_level=0, n_frames=6):
    """
    Show side-by-side: raw frame | detections (green=pred, red=GT missed).
    """
    env = create_environment(
        max_cycles=100,
        render_mode="rgb_array",
        distortion_level=distortion_level,
    )
    env.reset(seed=999)

    collected = []
    first_agent = env.possible_agents[0]

    for agent in env.agent_iter():
        if len(collected) >= n_frames:
            break
        obs, reward, term, trunc, info = env.last()
        done = term or trunc
        env.step(None if done else 0)

        if agent != first_agent:
            continue

        gt_boxes   = get_zombie_boxes(env)
        raw_frame  = env.render()
        inp        = preprocess_obs(raw_frame)
        with torch.no_grad():
            preds = model(inp)
        pred_boxes = decode_detections(preds, conf_threshold=CONF_THR,
                                       orig_w=ORIG_W, orig_h=ORIG_H)

        # only collect frames that have at least one zombie
        if len(gt_boxes) > 0:
            collected.append((raw_frame, pred_boxes, gt_boxes))

    env.close()

    if not collected:
        print("No zombie frames collected for visualisation")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Zombie Detection — Distortion Level {distortion_level}\n"
                 f"Green = predicted   Red = missed GT", fontsize=13)

    for idx, (frame, preds, gts) in enumerate(collected[:6]):
        ax = axes[idx // 3][idx % 3]
        # frame is (720, 1280, 3) — downscale for display
        display = np.array(Image.fromarray(frame).resize((640, 360)))
        ax.imshow(display)
        scale_x, scale_y = 640/ORIG_W, 360/ORIG_H

        # draw predicted boxes in green
        for box in preds:
            x, y, w, h = box * [scale_x, scale_y, scale_x, scale_y]
            ax.add_patch(patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='lime', facecolor='none'
            ))

        # draw GT boxes — green if matched, red if missed
        matched_gt = set()
        for pb in preds:
            for gi, gb in enumerate(gts):
                if gi not in matched_gt and box_iou(pb, gb) >= MATCH_IOU:
                    matched_gt.add(gi)

        for gi, gb in enumerate(gts):
            x, y, w, h = gb * [scale_x, scale_y, scale_x, scale_y]
            color = 'red' if gi not in matched_gt else 'lime'
            ax.add_patch(patches.Rectangle(
                (x, y), w, h,
                linewidth=1.5, edgecolor=color,
                facecolor='none', linestyle='--'
            ))

        tp_here = len(matched_gt)
        fn_here = len(gts) - tp_here
        ax.set_title(f"Found {tp_here}/{len(gts)} zombies", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    fname = f"detection_viz_level{distortion_level}.png"
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.show()
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    model = load_model(CHECKPOINT)

    # ── 1. quantitative evaluation across all levels ──
    print("\n" + "="*60)
    print("QUANTITATIVE EVALUATION — all distortion levels")
    print("="*60)
    print(f"{'Level':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} "
          f"{'TP':>6} {'FP':>6} {'FN':>6} {'All found%':>12}")
    print("-"*60)

    results = []
    for level in range(6):
        r = evaluate_level(model, level)
        results.append(r)
        status = " ✓" if r["recall"] >= 0.75 else " ✗ BELOW 75%"
        print(f"  {level:<6} {r['precision']:>10.3f} {r['recall']:>8.3f} "
              f"{r['f1']:>8.3f} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6} "
              f"{r['frames_all_found_%']:>10.1f}%{status}")

    # ── 2. summary plot ──
    levels     = [r["level"]     for r in results]
    precisions = [r["precision"] for r in results]
    recalls    = [r["recall"]    for r in results]
    f1s        = [r["f1"]        for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(levels, precisions, 'o-', color='#3498db', linewidth=2, label='Precision')
    ax.plot(levels, recalls,    's-', color='#2ecc71', linewidth=2, label='Recall')
    ax.plot(levels, f1s,        '^-', color='#e74c3c', linewidth=2, label='F1')
    ax.axhline(y=0.75, color='gray', linestyle='--', linewidth=1, label='75% target')
    ax.set_xlabel("Distortion Level", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Zombie Detection Performance by Distortion Level", fontsize=13)
    ax.set_xticks(levels)
    ax.set_xticklabels([f"Level {l}" for l in levels], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("detection_performance.png", dpi=150, bbox_inches='tight')
    print("\nSaved: detection_performance.png")
    plt.show()
    plt.close()

    # ── 3. visual inspection ──
    print("\nGenerating visual inspection plots...")
    for level in [0, 2, 5]:    # clean, medium, hardest
        visualise_detections(model, distortion_level=level)

    # ── 4. final verdict ──
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    level0_recall = results[0]["recall"]
    if level0_recall >= 0.75:
        print(f"  ✓ Level 0 recall = {level0_recall:.3f} — meets the 75% requirement")
    else:
        print(f"  ✗ Level 0 recall = {level0_recall:.3f} — BELOW 75% requirement")
        print(f"    → consider: more training data, more epochs, lower conf_threshold")


if __name__ == "__main__":
    main()