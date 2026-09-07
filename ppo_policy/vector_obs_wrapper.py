"""
Training-time wrapper that converts pixel observations into a 48-dim feature
vector by reading game state directly from env.unwrapped.

Authorized by professor's clarification:
  "During training, you can use everything you want. We only check appropriate
   access during inference."

At SUBMISSION time, the same vector format is constructed using:
  - CNN-detected zombies (from pixels — required by professor)
  - env.agent_list for archer state of BOTH archers (explicitly allowed)
"""
import numpy as np
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper

SCREEN_W = 1280
SCREEN_H = 720
MAX_ZOMBIES = 8
# features: 4 (archer 0), 4 (archer 1), 40 (zombies)
VECTOR_DIM = 4 + 4 + MAX_ZOMBIES * 5


def build_vector(my_archer, teammate_archer, zombie_positions):

    vec = np.zeros(VECTOR_DIM, dtype=np.float32)

    # Self state
    vec[0] = my_archer.rect.centerx / SCREEN_W
    vec[1] = my_archer.rect.centery / SCREEN_H
    vec[2] = float(my_archer.direction[0])
    vec[3] = float(my_archer.direction[1])

    # Teammate state (zeros if missing — graceful degradation)
    if teammate_archer is not None:
        vec[4] = teammate_archer.rect.centerx / SCREEN_W
        vec[5] = teammate_archer.rect.centery / SCREEN_H
        vec[6] = float(teammate_archer.direction[0])
        vec[7] = float(teammate_archer.direction[1])

    # Zombies sorted by y descending (most threatening first), top MAX_ZOMBIES
    sorted_z = sorted(zombie_positions, key=lambda z: -z[1])
    for i in range(min(len(sorted_z), MAX_ZOMBIES)):
        zx, zy, zw, zh = sorted_z[i]
        vec[8 + i * 5]     = zx / SCREEN_W
        vec[8 + i * 5 + 1] = zy / SCREEN_H
        vec[8 + i * 5 + 2] = zw / SCREEN_W
        vec[8 + i * 5 + 3] = zh / SCREEN_H
        # Presence flag: to distinguish between existing zombies and empty slots.
        vec[8 + i * 5 + 4] = 1.0

    # Final clipping of vector values to [-2,2] for training stability
    return np.clip(vec, -2.0, 2.0)


class VectorObsWrapper(BaseWrapper):
    """
    Training-time wrapper. Reads zombies from env.zombie_list (privileged) and
    both archers from env.agent_list. Returns a 32-dim float32 vector.
    """

    # Declare to PettingZoo the feature vector.
    def observation_space(self, agent):
        return spaces.Box(
            low=-2.0, high=2.0,
            shape=(VECTOR_DIM,),
            dtype=np.float32,
        )

    def observe(self, agent):
        try:
            game = self.env.unwrapped
            idx = game.agent_name_mapping[agent]
            my_archer = game.agent_list[idx]

            # Find teammate (the other archer in agent_list)
            teammate_archer = None
            for j, other in enumerate(game.agent_list):
                if j != idx:
                    teammate_archer = other
                    break

            zombie_positions = [
                (z.rect.centerx, z.rect.centery, z.rect.width, z.rect.height)
                for z in game.zombie_list
            ]
            return build_vector(my_archer, teammate_archer, zombie_positions)
        except Exception:
            return np.zeros(VECTOR_DIM, dtype=np.float32)