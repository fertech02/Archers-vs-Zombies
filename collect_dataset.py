import os
import pickle
import numpy as np
from PIL import Image

from utils import create_environment


def get_zombie_boxes(env):
    """Extract zombie bounding boxes from the game's sprite list, sorted by x."""
    game = env.unwrapped
    boxes = []
    for z in game.zombie_list:
        r = z.rect
        boxes.append([r.x, r.y, r.width, r.height])
    boxes.sort(key=lambda b: b[0])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def _scale_boxes(boxes, orig_hw, new_wh):
    """Scale [x, y, w, h] boxes from original pixel space to resized frame space."""
    if boxes.size == 0:
        return boxes
    sx = new_wh[0] / orig_hw[1]   # orig_hw = (H, W)
    sy = new_wh[1] / orig_hw[0]
    scaled = boxes.copy()
    scaled[:, [0, 2]] *= sx        # x, width
    scaled[:, [1, 3]] *= sy        # y, height
    return scaled


def _save_chunk(save_dir, chunk_idx, frames, labels):
    tag = f"{chunk_idx:04d}"
    np.save(os.path.join(save_dir, f"frames_{tag}.npy"), np.array(frames, dtype=np.uint8))
    with open(os.path.join(save_dir, f"labels_{tag}.pkl"), "wb") as f:
        pickle.dump(labels, f)


def load_dataset(save_dir="zombie_dataset"):
    """Load all chunks and return (frames, labels) arrays."""
    import glob
    frame_files = sorted(glob.glob(os.path.join(save_dir, "frames_*.npy")))
    label_files = sorted(glob.glob(os.path.join(save_dir, "labels_*.pkl")))
    frames = np.concatenate([np.load(f) for f in frame_files], axis=0)
    labels = []
    for f in label_files:
        with open(f, "rb") as fh:
            labels.extend(pickle.load(fh))
    return frames, labels


def collect(n_episodes=200, max_steps=300, save_dir="zombie_dataset",
            frame_size=(320, 180), save_every=10, distortion_level=None):
    """
    frame_size       : (width, height) to resize frames — reduces per-frame memory 16x
                       vs the native 1280x720.
    save_every       : flush a chunk to disk every N episodes to keep RAM bounded.
    distortion_level : None for random levels (0-5) per episode, or an int to force a specific level.
    """
    os.makedirs(save_dir, exist_ok=True)
    frames_buf, labels_buf = [], []
    chunk_idx = 0
    total_frames = 0
    
    # Initialize env outside the loop to prevent reference errors
    env = None

    for ep in range(n_episodes):
        # ALWAYS close the previous environment to prevent memory leaks
        if env is not None:
            env.close()

        # Randomize the distortion level if not explicitly provided
        level = np.random.randint(0, 6) if distortion_level is None else distortion_level

        env = create_environment(
            max_cycles=max_steps,
            render_mode="rgb_array",
            distortion_level=level,
        )

        env.reset(seed=ep)
        first_agent = env.possible_agents[0]

        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            done = term or trunc
            env.step(None if done else 0)
            if agent == first_agent:
                raw = env.render()                              # (H, W, 3) uint8
                frame = np.array(
                    Image.fromarray(raw).resize(frame_size, Image.BILINEAR),
                    dtype=np.uint8,
                )
                boxes = _scale_boxes(get_zombie_boxes(env), raw.shape[:2], frame_size)
                frames_buf.append(frame)
                labels_buf.append(boxes)
                total_frames += 1

        if (ep + 1) % save_every == 0:
            _save_chunk(save_dir, chunk_idx, frames_buf, labels_buf)
            print(f"Episode {ep+1}/{n_episodes} (Level {level}) — {total_frames} frames collected")
            frames_buf, labels_buf = [], []
            chunk_idx += 1

    if frames_buf:
        _save_chunk(save_dir, chunk_idx, frames_buf, labels_buf)
        chunk_idx += 1

    # Catch the final closure after the loop finishes
    if env is not None:
        env.close()
        
    print(f"Saved {total_frames} frames across {chunk_idx} chunk file(s) in {save_dir}/")


if __name__ == "__main__":
    collect()