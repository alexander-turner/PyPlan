from agents import *
from heuristics import rollout_heuristic
from simulators import pacman_sim

"""
Policy switching carries with it the theoretical guarantee that at each juncture, it will perform at least as well as 
the best policy in its set. However, observe that when an agent has a low number of pulls (and thus we can't have high 
confidence in its conclusions), it can be hard for the policy switching agent to estimate the policy's value. 
This makes it vulnerable to poor decision-making.
"""

if __name__ == '__main__':
    rand_agent = random_agent.RandomAgentClass()

    u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=1, num_pulls=100, policy=rand_agent)
    nested_u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=1, num_pulls=10, policy=u_ro)

    e_ro = e_rollout_agent.ERolloutAgentClass(depth=1, num_pulls=100, epsilon=0.5, policy=rand_agent)

    h = rollout_heuristic.RolloutHeuristicClass(rollout_policy=rand_agent, width=1, depth=10)
    ss = sparse_sampling_agent.SparseSamplingAgentClass(depth=2, pulls_per_node=20, heuristic=h)

    switch_agent = policy_switch_agent.PolicySwitchAgentClass(num_pulls=10, policies=[u_ro, ss])

    pacman = pacman_sim.PacmanStateClass(layout_repr='testClassic', use_graphics=True)
    pacman.run(agents=[u_ro, ss, switch_agent], num_trials=10, do_render=False)

    bad_ss = sparse_sampling_agent.SparseSamplingAgentClass(depth=3, pulls_per_node=5, heuristic=h)
    bad_ss.agentname += ' (bad)'
    bad_switch_agent = policy_switch_agent.PolicySwitchAgentClass(num_pulls=10, policies=[u_ro, bad_ss])
    bad_switch_agent.agentname += ' (bad)'

    pacman.run(agents=[u_ro, ss, bad_ss, switch_agent, bad_switch_agent], num_trials=10, do_render=False)
