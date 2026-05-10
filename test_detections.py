import argparse
import random
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import create_environment
from zombie_detection.cnn import ZombieCNN
from zombie_detection.preprocessing import decode_detections, preprocess_obs

parser = argparse.ArgumentParser()
parser.add_argument('--distortion', type=int, default=0)
parser.add_argument('--threshold', type=float, default=0.7)
parser.add_argument('--steps', type=int, default=80)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--out', type=str, default='detections_test.png')
args = parser.parse_args()

model = ZombieCNN(input_shape=(3, 90, 160))
model.load_state_dict(torch.load('zombie_detection/zombie_cnn.pth', map_location='cpu'))
model.eval()

env = create_environment(distortion_level=args.distortion, render_mode='rgb_array', max_cycles=1000)
env.reset(seed=args.seed)

frame = None

for i, agent in enumerate(env.agent_iter()):
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        env.step(None)
        continue
    action = 4 if i % 2 == 0 else random.choice([1, 2, 3])
    env.step(action)
    if i == args.steps:
        frame = env.render()
        break

env.close()

t = preprocess_obs(frame).cpu()
with torch.no_grad():
    preds = model(t)
best_boxes = decode_detections(preds, conf_threshold=args.threshold, orig_w=frame.shape[1], orig_h=frame.shape[0])
best_count = len(best_boxes)
best_frame = frame

fig, ax = plt.subplots(1, figsize=(16, 9))
ax.imshow(best_frame)
ax.set_title(f'distortion={args.distortion} | threshold={args.threshold} | {best_count} detections')
for b in best_boxes:
    ax.add_patch(patches.Rectangle((b[0], b[1]), b[2], b[3], linewidth=2, edgecolor='lime', facecolor='none'))
plt.savefig(args.out, dpi=80, bbox_inches='tight')
print(f'Saved {args.out} — {best_count} detections')
