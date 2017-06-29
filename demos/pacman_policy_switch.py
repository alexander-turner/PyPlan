from agents import *
from heuristics import rollout_heuristic
from simulators import pacman_sim, connect4_sim
from demos import simulate

"""
Policy switching carries with it the theoretical guarantee that at each juncture, it will perform at least as well as 
the best policy in its set. However, observe tha
"""

if __name__ == '__main__':
    rand_agent = random_agent.RandomAgentClass()

    u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=1, num_pulls=100, policy=rand_agent)
    nested_u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=1, num_pulls=10, policy=u_ro)
    e_ro = e_rollout_agent.ERolloutAgentClass(depth=1, num_pulls=100, epsilon=0.5, policy=rand_agent)
    h = rollout_heuristic.RolloutHeuristicClass(rollout_policy=rand_agent, width=1, depth=10)
    ss = sparse_sampling_agent.SparseSamplingAgentClass(depth=2, pulls_per_node=20, heuristic=h)

    switch_agent = policy_switch_agent.PolicySwitchAgentClass(num_pulls=10, policies=[u_ro, nested_u_ro, ss])

    pacman = pacman_sim.PacmanStateClass(layout_repr='testClassic', use_graphics=True)
    pacman.run(agents=[u_ro, nested_u_ro, ss, switch_agent], num_trials=3)
