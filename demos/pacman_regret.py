from agents import *
from heuristics import rollout_heuristic
from simulators import pacman_sim

"""
The UCB rollout agent attempts to minimize cumulative regret over its pull budget. Unlike uniform rollout, it doesn't 
spend much time on non-promising arms.
"""

if __name__ == '__main__':
    rand_agent = random_agent.RandomAgentClass()
    h1 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=rand_agent, width=1, depth=10)

    u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=1, num_pulls=100, policy=rand_agent)
    ucb_ro = ucb_rollout_agent.UCBRolloutAgentClass(depth=1, num_pulls=100, c=1.0, policy=rand_agent)

    pacman = pacman_sim.PacmanStateClass(layout_repr='testClassic', use_graphics=True)
    pacman.run(agents=[u_ro, e_ro, ucb_ro], num_trials=15, show_moves=False)
