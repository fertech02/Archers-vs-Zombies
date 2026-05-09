"""Template of your submission file for Task 3 (multi agent KAZ)."""
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
from zombie_detection.rllib_model import SimpleKAZModel

_HERE = Path(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH  = _HERE / "zombie_detection" / "zombie_cnn.pth"
_POLICY_PATH = _HERE / "policy.pth"

_ZOMBIE_W_NORM = 7.25 / 320
_ZOMBIE_H_NORM = 7.75 / 180

# Tells the evaluation harness how to build the env
ENV_SETTINGS = {
    "distortion_level": 0,
}


class CustomWrapper(BaseWrapper):
    """Identity wrapper — env_settings already give the obs shape the policy expects."""
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.observation_space(agent)

    def observe(self, agent: AgentID) -> ObsType | None:
        return self.env.observe(agent)


class CustomPredictFunction(Callable):
    """Loads the trained PPO policy weights and predicts actions."""

    def __init__(self, env: gymnasium.Env):
        agent_id = list(env.possible_agents)[0] if hasattr(env, "possible_agents") else "archer_0"

        self.model = SimpleKAZModel(
            obs_space=env.observation_space(agent_id),
            action_space=env.action_space(agent_id),
            num_outputs=env.action_space(agent_id).n,
            model_config={"custom_model_config": {
                "cnn_checkpoint": str(_MODEL_PATH),
            }},
            name="kaz_simple",
        )

        if _POLICY_PATH.exists():
            self.model.load_state_dict(
                torch.load(str(_POLICY_PATH), map_location="cpu"), strict=True
            )

        self.model.eval()

    def __call__(self, observation, agent, *args, **kwargs):
        obs_t = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model({"obs": obs_t}, [], None)
        return int(torch.argmax(logits, dim=-1).item())


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

        boxes = decode_detections(preds, conf_threshold=0.7, orig_w=orig_w, orig_h=orig_h)
        if len(boxes) > 0:
            boxes[:, 2] = _ZOMBIE_W_NORM * orig_w
            boxes[:, 3] = _ZOMBIE_H_NORM * orig_h
        return boxes
