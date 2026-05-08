"""
reward_wrapper.py
-----------------
Reward shaping for KAZ training only.
Never used in submission/evaluation (tournament uses default PettingZoo rewards).
"""
import math
from pettingzoo.utils import BaseWrapper

SCREEN_W = 1280
SCREEN_H = 720
ATTACK_ACTION = 4


class ShapedRewardWrapper(BaseWrapper):

    def __init__(self, env):
        super().__init__(env)
        # Track the last action per agent so we can check it in last()
        # (attacking flag on sprite resets too fast to be reliable)
        self._last_action = {}

    def step(self, action):
        agent = self.env.agent_selection
        self._last_action[agent] = action
        self.env.step(action)

    def last(self):
        obs, reward, term, trunc, info = self.env.last()

        # 1. Amplify sparse kill/death signals
        if reward > 0:
            reward *= 10.0   # +10 for a kill
        elif reward < 0:
            reward *= 5.0    # -5 for dying

        if term or trunc:
            return obs, reward, term, trunc, info

        # 2. Dense shaping
        reward += self._compute_shaping()
        return obs, reward, term, trunc, info

    def _compute_shaping(self) -> float:
        try:
            game = self.env.unwrapped
            agent = self.env.agent_selection
            zombies = list(game.zombie_list)

            idx = game.agent_name_mapping[agent]
            my_archer = game.agent_list[idx]

            my_x = my_archer.rect.centerx
            my_y = my_archer.rect.centery

            # direction is a unit vector updated by the game engine each step
            hx = my_archer.direction[0]
            hy = my_archer.direction[1]

        except Exception:
            return 0.0

        total = 0.0

        if zombies:
            # --- SECTORS OF FIRE ---
            # archer_0 owns the left half, archer_1 owns the right half.
            # If own zone is empty, help the other side.
            if agent == "archer_0":
                my_zone = [z for z in zombies if z.rect.centerx < SCREEN_W / 2]
            else:
                my_zone = [z for z in zombies if z.rect.centerx >= SCREEN_W / 2]

            active_zombies = my_zone if my_zone else zombies

            # Most dangerous zombie in active zone = lowest on screen (highest y)
            target = max(active_zombies, key=lambda z: z.rect.y)
            tx = target.rect.centerx
            ty = target.rect.centery

            dx = tx - my_x
            dy = ty - my_y
            mag = math.hypot(dx, dy)

            if mag > 0:
                # Dot product aiming: how well is the archer aimed at the target?
                # 1.0 = perfect, 0.0 = 90deg off, -1.0 = facing away
                aim = (hx * dx + hy * dy) / mag
                total += 0.03 * max(0.0, aim)

                # Attack bonus: reward firing when well-aimed
                # Uses last_action because attacking flag resets before last() is called
                last_action = self._last_action.get(agent, None)
                if last_action == ATTACK_ACTION and aim > 0.7:
                    total += 0.05

            # Urgency: small penalty as zombie approaches bottom
            total -= 0.005 * (ty / SCREEN_H)

        return total