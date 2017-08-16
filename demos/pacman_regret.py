from agents import *
from dealers import pacman_dealer

"""
The UCB rollout agent attempts to minimize cumulative regret over its pull budget. Unlike uniform rollout, it doesn't 
spend much time on non-promising arms.
"""

if __name__ == '__main__':
    u_ro = uniform_rollout_agent.UniformRolloutAgent(depth=1, num_pulls=100)
    ucb_ro = ucb_rollout_agent.UCBRolloutAgent(depth=1, num_pulls=100, c=1.0)

    pacman = pacman_dealer.Dealer(layout_repr='testClassic')
    pacman.run(agents=[u_ro, ucb_ro], num_trials=15)
