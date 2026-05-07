"""
Zombie detection evaluation.
Run: python test_zombie_detector.py

Set IS_MY_MODEL = True  if testing YOUR retrained model (trained with action=0)
Set IS_MY_MODEL = False if testing TEAMMATE's original model (trained with random actions)
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
from collect_dataset import get_zombie_boxes
from utils import create_environment

# ── CONFIG — only edit this block ─────────────────────────────────────────────

IS_MY_MODEL    = False                          # True = your model, False = teammate's
CHECKPOINT     = "zombie_detection/zombie_cnn.pth"
CONF_THR       = 0.80
IOU_THR        = 0.40
MATCH_IOU      = 0.50
N_EVAL_EPS     = 5
MAX_STEPS      = 200
ORIG_W         = 1280
ORIG_H         = 720

# ── END CONFIG ─────────────────────────────────────────────────────────────────


def load_model(checkpoint):
    model = ZombieCNN(input_shape=(3, 90, 160))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    print(f"Loaded: {checkpoint}")
    print(f"Mode:   {'MY retrained model (action=0)' if IS_MY_MODEL else 'TEAMMATE original model (random actions)'}")
    return model


def box_iou(a, b):
    ax2 = a[0] + a[2];  ay2 = a[1] + a[3]
    bx2 = b[0] + b[2];  by2 = b[1] + b[3]
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(ax2,  bx2);   iy2 = min(ay2,  by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred_boxes, gt_boxes):
    """Returns (tp, fp, fn)."""
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
        if best_iou >= MATCH_IOU:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def evaluate_level(model, distortion_level):
    env = create_environment(
        max_cycles=MAX_STEPS,
        render_mode="rgb_array",
        distortion_level=distortion_level,
    )

    total_tp = total_fp = total_fn = 0
    frames_evaluated = 0
    frames_all_found = 0
    first_agent = env.possible_agents[0]

    for ep in range(N_EVAL_EPS):
        env.reset(seed=ep + 999)

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            done = term or trunc
            env.step(None if done else env.action_space(agent).sample())

            # only evaluate on first agent's turn to avoid double-counting
            if agent != first_agent:
                continue

            gt_boxes = get_zombie_boxes(env)

            raw_frame = env.render()
            inp = preprocess_obs(raw_frame)
            with torch.no_grad():
                preds = model(inp)
            pred_boxes = decode_detections(
                preds,
                conf_threshold=CONF_THR,
                iou_threshold=IOU_THR,
                orig_w=ORIG_W,
                orig_h=ORIG_H,
            )

            tp, fp, fn = match_boxes(pred_boxes, gt_boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            frames_evaluated += 1
            if fn == 0:
                frames_all_found += 1

    env.close()

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    all_found = frames_all_found / frames_evaluated * 100 if frames_evaluated > 0 else 0.0

    return {
        "level":     distortion_level,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp":        total_tp,
        "fp":        total_fp,
        "fn":        total_fn,
        "all_found": all_found,
        "n_frames":  frames_evaluated,
    }


def visualise_detections(model, distortion_level=0, n_frames=6):
    env = create_environment(
        max_cycles=300,
        render_mode="rgb_array",
        distortion_level=distortion_level,
    )
    env.reset(seed=777)
    collected = []
    first_agent = env.possible_agents[0]

    for agent in env.agent_iter():
        if len(collected) >= n_frames:
            break
        obs, reward, term, trunc, info = env.last()
        done = term or trunc
        env.step(None if done else env.action_space(agent).sample())

        if agent != first_agent:
            continue

        gt_boxes = get_zombie_boxes(env)
        if len(gt_boxes) == 0:
            continue

        raw_frame = env.render()
        inp = preprocess_obs(raw_frame)
        with torch.no_grad():
            preds = model(inp)
        pred_boxes = decode_detections(preds, conf_threshold=CONF_THR,
                                       orig_w=ORIG_W, orig_h=ORIG_H)
        collected.append((raw_frame, pred_boxes, gt_boxes))

    env.close()

    if not collected:
        print(f"  No zombie frames collected for level {distortion_level}")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"Level {distortion_level} — green box = predicted   red dashed = missed zombie",
        fontsize=12
    )

    for idx, (frame, preds, gts) in enumerate(collected[:6]):
        ax = axes[idx // 3][idx % 3]
        display = np.array(Image.fromarray(frame).resize((640, 360)))
        ax.imshow(display)
        sx, sy = 640 / ORIG_W, 360 / ORIG_H

        # figure out which GT boxes were matched
        matched_gt = set()
        for pb in preds:
            for gi, gb in enumerate(gts):
                if gi not in matched_gt and box_iou(pb, gb) >= MATCH_IOU:
                    matched_gt.add(gi)

        # draw predictions in green
        for box in preds:
            x, y, w, h = box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy
            ax.add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none'
            ))

        # draw GT — lime if found, red if missed
        for gi, gb in enumerate(gts):
            x, y, w, h = gb[0]*sx, gb[1]*sy, gb[2]*sx, gb[3]*sy
            color = 'lime' if gi in matched_gt else 'red'
            ax.add_patch(patches.Rectangle(
                (x, y), w, h,
                linewidth=1.5, edgecolor=color,
                facecolor='none', linestyle='--'
            ))

        ax.set_title(f"Found {len(matched_gt)}/{len(gts)}", fontsize=10)
        ax.axis('off')

    # hide unused subplots
    for idx in range(len(collected), 6):
        axes[idx // 3][idx % 3].axis('off')

    plt.tight_layout()
    fname = f"detection_viz_level{distortion_level}.png"
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def main():
    model = load_model(CHECKPOINT)

    print("\n" + "="*65)
    print("QUANTITATIVE EVALUATION — all distortion levels")
    print("="*65)
    print(f"  conf_threshold={CONF_THR}   match_iou={MATCH_IOU}   episodes={N_EVAL_EPS}")
    print("-"*65)
    print(f"{'Level':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} "
          f"{'TP':>6} {'FP':>6} {'FN':>6} {'Frames':>7} {'All%':>7}")
    print("-"*65)

    results = []
    for level in range(6):
        r = evaluate_level(model, level)
        results.append(r)
        status = "✓" if r["recall"] >= 0.75 else "✗"
        print(f"  {level:<6} {r['precision']:>10.3f} {r['recall']:>8.3f} "
              f"{r['f1']:>8.3f} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6} "
              f"{r['n_frames']:>7} {r['all_found']:>6.1f}% {status}")

    # ── performance plot ───────────────────────────────────────────────────────
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
    plt.close()

    # ── visual inspection for levels 0, 2, 5 ──────────────────────────────────
    print("\nGenerating visual inspection...")
    for level in [0, 2, 5]:
        visualise_detections(model, distortion_level=level)

    # ── final verdict ──────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("FINAL VERDICT")
    print("="*65)
    r0 = results[0]
    print(f"  Level 0 — Precision={r0['precision']:.3f}  Recall={r0['recall']:.3f}  F1={r0['f1']:.3f}")
    if r0["recall"] >= 0.75:
        print(f"  ✓ Recall meets the 75% assignment requirement")
    else:
        print(f"  ✗ Recall is below 75% — try lowering CONF_THR or retraining")

    print(f"\n  Tip: if precision is low → raise CONF_THR")
    print(f"       if recall is low   → lower CONF_THR")
    print(f"       current CONF_THR   = {CONF_THR}")


if __name__ == "__main__":
    main()