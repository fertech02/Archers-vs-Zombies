python3 -c "
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from submission import CustomZombieDetectorFunction
from utils import create_environment

obs_flat = np.load('observation_data/example_obs.npy')
obs = obs_flat.reshape(720, 1280, 3)
zombies_gt = np.load('observation_data/example_zombies.npy')

env = create_environment(distortion_level=0)
detector = CustomZombieDetectorFunction(env)
boxes_pred = detector(obs_flat)

fig, ax = plt.subplots(1, figsize=(16, 9))
ax.imshow(obs)

for z in zombies_gt:
    ax.add_patch(patches.Rectangle((z[0], z[1]), z[2], z[3], linewidth=3, edgecolor='red', facecolor='none', label='ground truth'))

for b in boxes_pred:
    ax.add_patch(patches.Rectangle((b[0], b[1]), b[2], b[3], linewidth=1, edgecolor='lime', facecolor='none', alpha=0.5, label='predicted'))

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='red', label='ground truth'), Patch(color='lime', label='predicted')])
plt.savefig('detections.png', dpi=80, bbox_inches='tight')
print('saved detections.png - red=real zombies, green=predictions')
"