"""
reward_wrapper.py
-----------------
Reward shaping wrapper for training only.
NEVER used in submission.py — tournament uses default PettingZoo rewards.

Verified attribute names from KAZ source:
  game.zombie_list        → pygame.sprite.Group of zombies
  game.archer_list        → pygame.sprite.Group of archers
  game.agent_list         → list of all agents indexed by agent_name_mapping
  game.agent_name_mapping → dict {agent_name_str: index}
  sprite.rect.x           → x position in pixels
  sprite.rect.y           → y position in pixels
  SCREEN_WIDTH  = 1280
  SCREEN_HEIGHT = 720
"""

from pettingzoo.utils import BaseWrapper

SCREEN_W = 1280
SCREEN_H = 720


class ShapedRewardWrapper(BaseWrapper):
    """
    Wraps the KAZ AEC environment and adds reward shaping terms on top of
    the default PettingZoo rewards (+1 kill, -1 die).

    Shaping terms (all small relative to +1 kill reward):
      - Danger penalty:    proportional to how close the nearest zombie is
                           to the bottom of the screen. Teaches urgency.
      - Survival bonus:    tiny positive signal every step alive.
                           Teaches the agent that staying alive is good
                           continuously, not just when it dies.
      - Zombie count:      small penalty per zombie currently alive.
                           Encourages clearing the screen efficiently.
      - Zone bonus:        archer_0 gets bonus for being in left half,
                           archer_1 for right half. Teaches spatial division.
      - Crowding penalty:  both archers penalized when too close together.
                           Reinforces the zone separation.

    Intercepted in last() — the correct AEC pattern.
    step() is NOT overridden — we never touch it.
    """

    def last(self):
        obs, reward, term, trunc, info = self.env.last()

        # don't shape terminal steps — agent is already done
        if term or trunc:
            return obs, reward, term, trunc, info

        shaping = self._compute_shaping()
        return obs, reward + shaping, term, trunc, info

    # ── shaping computation ────────────────────────────────────────────────────

    def _compute_shaping(self) -> float:
        try:
            game     = self.env.unwrapped
            agent    = self.env.agent_selection
            zombies  = list(game.zombie_list)
            archers  = list(game.archer_list)
        except Exception:
            return 0.0

        total = 0.0

        # ── survival bonus ─────────────────────────────────────────────────────
        # tiny reward every step the agent is alive
        # teaches that being alive is continuously good, not just at death
        total += 0.005

        # ── danger penalty ─────────────────────────────────────────────────────
        # grows as the lowest zombie approaches the bottom line
        # teaches urgency — prioritize the most dangerous zombie
        if zombies:
            lowest_y  = max(z.rect.y for z in zombies)   # highest y = closest to bottom
            danger    = lowest_y / SCREEN_H               # normalized 0→1
            total    -= 0.05 * danger

        # ── zombie count penalty ───────────────────────────────────────────────
        # penalize having many zombies alive simultaneously
        # encourages efficient clearing rather than letting them accumulate
        total -= 0.005 * len(zombies)

        # ── zone coordination ──────────────────────────────────────────────────
        # get this agent's archer object
        try:
            idx        = game.agent_name_mapping[agent]
            my_archer  = game.agent_list[idx]
            my_x       = my_archer.rect.x
        except Exception:
            return total

        # archer_0 owns the left half, archer_1 owns the right half
        if agent == "archer_0":
            # bonus grows as archer moves toward left side
            zone_center = SCREEN_W * 0.25
        else:
            # archer_1 bonus grows toward right side
            zone_center = SCREEN_W * 0.75

        dist_from_zone = abs(my_x - zone_center) / SCREEN_W
        total += 0.01 * (1.0 - dist_from_zone)   # max +0.01 when perfectly in zone

        # ── crowding penalty ───────────────────────────────────────────────────
        # penalize both archers clustering together
        # reinforces spatial separation — if they're far apart, no penalty
        if len(archers) >= 2:
            # get the other archer's x position
            try:
                other_name  = "archer_1" if agent == "archer_0" else "archer_0"
                other_idx   = game.agent_name_mapping[other_name]
                other_x     = game.agent_list[other_idx].rect.x
                dist        = abs(my_x - other_x) / SCREEN_W   # normalized 0→1
                if dist < 0.15:
                    # within 15% of screen width — they're too close
                    total -= 0.02
            except Exception:
                pass

        return total