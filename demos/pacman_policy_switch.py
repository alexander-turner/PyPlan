from agents import *
from dealers import pacman_dealer

"""
Policy switching carries with it the theoretical guarantee that at each time step, it will perform at least as well as 
the best policy in its set. However, a certain number of pulls is required to accurately estimate the value of a given 
policy. If an insufficient pull budget is provided, policy switching may do somewhat worse than its best constituent
policy.
"""
if __name__ == '__main__':
    u_ro = uniform_rollout_agent.UniformRolloutAgent(depth=1, num_pulls=100)
    ss = sparse_sampling_agent.SparseSamplingAgent(depth=2, pulls_per_node=20)
    switching_agent = policy_switching_agent.PolicySwitchingAgent(depth=3, num_pulls=15, policies=[u_ro, ss])

    pacman = pacman_dealer.Dealer(layout_repr='testClassic')
    pacman.run(agents=[u_ro, ss, switching_agent], num_trials=1, multiprocess_mode='')
