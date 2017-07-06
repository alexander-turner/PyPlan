from agents import *
from heuristics import rollout_heuristic
from simulators import pacman_sim

"""
Policy switching carries with it the theoretical guarantee that at each time step, it will perform at least as well as 
the best policy in its set. However, when a constituent policy has a low number of pulls, the policy-switching agent 
needs more pulls to maintain the same confidence level in its estimations.
"""

if __name__ == '__main__':
    rand_agent = random_agent.RandomAgentClass()
    h1 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=rand_agent, width=1, depth=10)

    u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=1, num_pulls=100, policy=rand_agent)

    ss = sparse_sampling_agent.SparseSamplingAgentClass(depth=2, pulls_per_node=20, heuristic=h1)

    bad_ss = sparse_sampling_agent.SparseSamplingAgentClass(depth=3, pulls_per_node=5, heuristic=h1)
    bad_ss.agent_name += ' (bad)'

    switch_agent = policy_switch_agent.PolicySwitchAgentClass(num_pulls=10, policies=[u_ro, ss])

    bad_switch_agent = policy_switch_agent.PolicySwitchAgentClass(num_pulls=10, policies=[u_ro, bad_ss])
    bad_switch_agent.agent_name += ' (bad)'

    pacman = pacman_sim.Dealer(layout_representation='testClassic')
    pacman.run(agents=[u_ro, ss, bad_ss, switch_agent, bad_switch_agent], num_trials=50)
