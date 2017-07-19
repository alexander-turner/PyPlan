from agents import *
from heuristics import rollout_heuristic
from simulators import pacman_sim

"""
Policy switching carries with it the theoretical guarantee that at each time step, it will perform at least as well as 
the best policy in its set. However, a certain number of pulls is required to accurately estimate the value of a given 
policy. If an insufficient pull budget is provided, policy switching may do somewhat worse than its best constituent
policy.
"""
if __name__ == '__main__':
    h1 = rollout_heuristic.RolloutHeuristicClass(width=1, depth=10)

    u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=1, num_pulls=100)
    ss = sparse_sampling_agent.SparseSamplingAgentClass(depth=2, pulls_per_node=20, heuristic=h1)
    switch_agent = policy_switch_agent.PolicySwitchAgentClass(depth=3, num_pulls=15, policies=[u_ro, ss])

    pacman = pacman_sim.Dealer(layout_representation='testClassic')
    pacman.run(agents=[u_ro, ss, switch_agent], num_trials=15)
