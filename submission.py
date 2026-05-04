"""Template of your submission file for Task 3 (multi agent KAZ).
"""
import os
from typing import Callable

import gymnasium
import torch
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

from zombie_detection.cnn import ZombieCNN
from zombie_detection.preprocessing import decode_detections, preprocess_obs

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "zombie_detection", "zombie_cnn.pth")


class CustomWrapper(BaseWrapper):
    """
    Wrapper to use to add state pre-processing (feature engineering)
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.observation_space(agent)

    def observe(self, agent: AgentID) -> ObsType | None:
        return self.env.observe(agent)


class CustomPredictFunction(Callable):
    """Function to use to load the trained model and predict the action"""

    def __init__(self, env: gymnasium.Env):
        self.env = env

    def __call__(self, observation, agent, *args, **kwargs):
        return self.env.action_space(agent).sample()


class CustomZombieDetectorFunction(Callable):
    """Function to use to load the trained model and predict where
    the zombies are.
    """

    def __init__(self, env: gymnasium.Env):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ZombieCNN(input_shape=(3, 90, 160))
        self.model.load_state_dict(torch.load(_MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, observation, *args, **kwargs):
        """Returns a matrix of shape (nb_zombies, nb_attributes), where
        the attributes are defining a rectangle with (x,y,width,height) and
        indicate where the zombies are. The zombies are ordered from most
        likely to least likely positions. The evaluation uses the first k
        items if there are k zombies on the screen.
        """
        if observation.ndim == 1:
            # flat (H*W*3,) uint8 — reshape assuming 16:9 and 3 channels
            n_pixels = observation.size // 3
            orig_h = int((n_pixels * 9 / 16) ** 0.5)
            orig_w = n_pixels // orig_h
            observation = observation.reshape(orig_h, orig_w, 3)
        orig_h, orig_w = observation.shape[:2]
        tensor = preprocess_obs(observation).to(self.device)

        with torch.no_grad():
            preds = self.model(tensor)

        boxes = decode_detections(preds, conf_threshold=0.5, orig_w=orig_w, orig_h=orig_h)
        return boxes  # (k, 4) array: [x, y, w, h] sorted by confidence
