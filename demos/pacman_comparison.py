from agents import *
from dealers import pacman_dealer
from demos import generate_regret_curves

"""

"""

if __name__ == '__main__':
    u_ro = uniform_rollout_agent.UniformRolloutAgent(depth=1, num_pulls=10)
    e_ro = e_rollout_agent.ERolloutAgent(depth=1, num_pulls=10)
    ucb_ro = ucb_rollout_agent.UCBRolloutAgent(depth=1, num_pulls=10, c=1.0)

    pacman = pacman_dealer.Dealer(layout_repr='testClassic')
    generate_regret_curves.generate_regret_curves(agents=[u_ro, e_ro, ucb_ro], pull_max=100, simulator=pacman)