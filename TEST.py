from utils import create_environment
import pprint

env = create_environment(distortion_level=0, render_mode=None, max_cycles=50)
env.reset(seed=0)
game = env.unwrapped

print('agent_list:')
pprint.pprint(game.agent_list)
for a in game.agent_list:
    pprint.pprint(vars(a))

print()
print('agent_selection:', env.agent_selection)
env.close()