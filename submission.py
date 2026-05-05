import os
from pathlib import Path
from typing import Callable

import gymnasium
import numpy as np
import torch
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

from zombie_detection.cnn import ZombieCNN
from zombie_detection.preprocessing import decode_detections, preprocess_obs
from zombie_detection.rllib_model import KAZVisionModel

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = _HERE / "zombie_detection" / "zombie_cnn.pth"
_CHECKPOINT_ROOT = _HERE / "results" / "ppo_kaz" / "kaz_ppo"

_ZOMBIE_W_NORM = 7.25 / 320
_ZOMBIE_H_NORM = 7.75 / 180

# Tells the evaluation harness how to build the env.
ENV_SETTINGS = {
    "frame_stack": 4,
    "distortion_level": 5,
}


def _find_latest_checkpoint() -> Path:
    candidates = sorted(
        _CHECKPOINT_ROOT.glob("PPO_kaz_*/checkpoint_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No PPO checkpoint under {_CHECKPOINT_ROOT}")
    return candidates[0].resolve()


class CustomWrapper(BaseWrapper):
    """Identity wrapper — env_settings already give the obs shape the policy expects."""

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.observation_space(agent)

    def observe(self, agent: AgentID) -> ObsType | None:
        return self.env.observe(agent)


class CustomPredictFunction(Callable):
    """Loads the trained PPO policy (old RLlib API stack) and predicts actions."""

    def __init__(self, env):
        from ray.rllib.models import ModelCatalog
        from ray.rllib.policy.policy import Policy

        ModelCatalog.register_custom_model("kaz_vision", KAZVisionModel)

        ckpt = _find_latest_checkpoint()
        policy_dir = ckpt / "policies" / "default_policy"
        loaded = Policy.from_checkpoint(str(policy_dir if policy_dir.is_dir() else ckpt))
        self.policy = loaded["default_policy"] if isinstance(loaded, dict) else loaded

    def __call__(self, observation, agent, *args, **kwargs):
        action, _, _ = self.policy.compute_single_action(observation, explore=False)
        return action


class CustomZombieDetectorFunction(Callable):
    """Pretrained ZombieCNN detection head."""

    def __init__(self, env: gymnasium.Env):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ZombieCNN(input_shape=(3, 90, 160))
        self.model.load_state_dict(torch.load(str(_MODEL_PATH), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, observation, *args, **kwargs):
        if observation.ndim == 1:
            n_pixels = observation.size // 3
            orig_h = int((n_pixels * 9 / 16) ** 0.5)
            orig_w = n_pixels // orig_h
            observation = observation.reshape(orig_h, orig_w, 3)
        orig_h, orig_w = observation.shape[:2]
        tensor = preprocess_obs(observation).to(self.device)

        with torch.no_grad():
            preds = self.model(tensor)

        boxes = decode_detections(preds, conf_threshold=0.5, orig_w=orig_w, orig_h=orig_h)
        if len(boxes) > 0:
            boxes[:, 2] = _ZOMBIE_W_NORM * orig_w
            boxes[:, 3] = _ZOMBIE_H_NORM * orig_h
        return boxes