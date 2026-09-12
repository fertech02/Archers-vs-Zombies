import os
import pickle
import random
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


def scale_boxes(boxes, orig_hw, new_wh):
    if boxes.size == 0:
        return boxes
    sx = new_wh[0] / orig_hw[1]
    sy = new_wh[1] / orig_hw[0]
    scaled = boxes.copy()
    scaled[:, [0, 2]] *= sx
    scaled[:, [1, 3]] *= sy
    return scaled


def save_chunk(save_dir, chunk_idx, frames, labels):
    tag = f"{chunk_idx:04d}"
    np.save(os.path.join(save_dir, f"frames_{tag}.npy"), np.array(frames, dtype=np.uint8))
    with open(os.path.join(save_dir, f"labels_{tag}.pkl"), "wb") as f:
        pickle.dump(labels, f)


def load_dataset(save_dir="zombie_dataset", return_chunks=False):
    """
    return chunks: also return, for each frame, the index of the chunk file it
    came from. A chunk is a whole group of episodes.
    """
    import glob
    frame_files = sorted(glob.glob(os.path.join(save_dir, "frames_*.npy")))
    label_files = sorted(glob.glob(os.path.join(save_dir, "labels_*.pkl")))

    chunks = []
    frame_arrays = []
    for i, f in enumerate(frame_files):
        arr = np.load(f)
        frame_arrays.append(arr)
        chunks.append(np.full(len(arr), i, dtype=np.int64))

    frames = np.concatenate(frame_arrays, axis=0)
    labels = []

    for f in label_files:
        with open(f, "rb") as fh:
            labels.extend(pickle.load(fh))

    if return_chunks:
        return frames, labels, np.concatenate(chunks)
    return frames, labels


def collect(n_episodes=300, max_steps=500, save_dir="zombie_dataset",
            frame_size=(320, 180), save_every=10, distortion_level=None):
    """
    max_steps        : episode length.
    frame_size       : (width, height) to resize frames.
    save_every       : flush a chunk to disk every N episodes to keep RAM bounded.
    distortion_level : None for random levels (0-5) per episode, or an int to force a specific level.
    """
    os.makedirs(save_dir, exist_ok=True)
    frames_buf, labels_buf = [], []
    chunk_idx = 0
    total_frames = 0

    env = None

    for ep in range(n_episodes):
        if env is not None:
            env.close()

        level = np.random.randint(0, 6) if distortion_level is None else distortion_level

        env = create_environment(
            max_cycles=max_steps,
            render_mode="rgb_array",
            distortion_level=level,
        )

        # Change the seed for each episode
        env.reset(seed=ep)
        first_agent = env.possible_agents[0]

        for agent in env.agent_iter():

            # Current state for the given agent
            obs, reward, term, trunc, info = env.last()
            done = term or trunc

            # Choose a random action to perform by the agent
            env.step(None if done else random.choice([1, 2, 3, 5]))
            if agent == first_agent:
                raw = env.render()
                frame = np.array(
                    Image.fromarray(raw).resize(frame_size, Image.BILINEAR),
                    dtype=np.uint8,
                )
                boxes = scale_boxes(get_zombie_boxes(env), raw.shape[:2], frame_size)
                frames_buf.append(frame)
                labels_buf.append(boxes)
                total_frames += 1

        if (ep + 1) % save_every == 0:
            save_chunk(save_dir, chunk_idx, frames_buf, labels_buf)
            print(f"Episode {ep+1}/{n_episodes} (Level {level}) — {total_frames} frames collected")
            frames_buf, labels_buf = [], []
            chunk_idx += 1

    if frames_buf:
        save_chunk(save_dir, chunk_idx, frames_buf, labels_buf)
        chunk_idx += 1

    if env is not None:
        env.close()

    print(f"Saved {total_frames} frames across {chunk_idx} chunk file(s) in {save_dir}/")


if __name__ == "__main__":
    collect()

