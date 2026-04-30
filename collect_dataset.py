"""
Collect (frame, zombie_bboxes) pairs by running random agents
and reading zombie positions directly from the game state.
"""
import os
import pickle
import numpy as np
from pettingzoo.butterfly import knights_archers_zombies_v10


def get_zombie_boxes(env):
    """Extract zombie bounding boxes from the game's sprite list."""
    game = env.unwrapped          # strip all wrappers
    boxes = []
    for z in game.zombie_list:    # pygame sprite group
        r = z.rect
        boxes.append([r.x, r.y, r.width, r.height])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def collect(n_episodes=100, max_steps=300, save_dir="zombie_dataset"):
    os.makedirs(save_dir, exist_ok=True)
    frames, labels = [], []

    env = knights_archers_zombies_v10.env(render_mode="rgb_array", max_cycles=max_steps)

    for ep in range(n_episodes):
        env.reset(seed=ep)
        step = 0

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            done = term or trunc
            env.step(None if done else env.action_space(agent).sample())

            # Only sample once per full round (after all agents acted)
            if agent == env.agents[0] if env.agents else False:
                frame = env.render()          # (H, W, 3) uint8
                boxes = get_zombie_boxes(env)
                frames.append(frame)
                labels.append(boxes)
                step += 1

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{n_episodes}  —  {len(frames)} frames collected")

    env.close()

    np.save(os.path.join(save_dir, "frames.npy"), np.array(frames))
    with open(os.path.join(save_dir, "labels.pkl"), "wb") as f:
        pickle.dump(labels, f)

    print(f"Saved {len(frames)} frames to {save_dir}/")


if __name__ == "__main__":
    collect()
