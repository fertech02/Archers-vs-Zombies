"""
Extract clean PyTorch weights from a Ray RLlib checkpoint into
clean_kaz_weights.pth (the file submission.py expects).

Usage:
    python extract_weights.py path/to/checkpoint_XXXXXX
"""
import os
import sys
from pathlib import Path

import torch
from ray.rllib.algorithms.algorithm import Algorithm

from zombie_detection.rllib_model import KAZVisionModel  # noqa: F401  (registers model)
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("kaz_vision", KAZVisionModel)


def main(ckpt_path: str):
    ckpt_path = Path(ckpt_path).resolve()
    out_path = Path(__file__).parent / "clean_kaz_weights.pth"

    print(f"Loading checkpoint: {ckpt_path}")
    algo = Algorithm.from_checkpoint(str(ckpt_path))
    policy = algo.get_policy()
    state_dict = policy.model.state_dict()

    torch.save(state_dict, str(out_path))
    print(f"Saved {len(state_dict)} tensors to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_weights.py <checkpoint_dir>")
        sys.exit(1)
    main(sys.argv[1])