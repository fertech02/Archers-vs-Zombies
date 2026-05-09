import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from zombie_detection.cnn import ZombieCNN
from zombie_detection.preprocessing import preprocess_obs
from utils import create_environment

model = ZombieCNN(input_shape=(3, 90, 160))
model.load_state_dict(torch.load("zombie_detection/zombie_cnn.pth", map_location="cpu"))
model.eval()

env = create_environment(max_cycles=300, render_mode="rgb_array", distortion_level=0)
env.reset(seed=42)

for agent in env.agent_iter():
    env.step(0)
    from collect_dataset import get_zombie_boxes
    boxes = get_zombie_boxes(env)
    if len(boxes) > 0:
        raw = env.render()
        inp = preprocess_obs(raw)
        with torch.no_grad():
            preds = model(inp)
        confs = preds[0, :, 0]
        print(f"Zombies in frame:  {len(boxes)}")
        print(f"Max confidence:    {confs.max().item():.4f}")
        print(f"Mean confidence:   {confs.mean().item():.4f}")
        print(f"Cells above 0.3:   {(confs > 0.3).sum().item()}")
        print(f"Cells above 0.5:   {(confs > 0.5).sum().item()}")
        print(f"Cells above 0.8:   {(confs > 0.8).sum().item()}")
        print(f"Cells above 0.9:   {(confs > 0.9).sum().item()}")
        break

env.close()