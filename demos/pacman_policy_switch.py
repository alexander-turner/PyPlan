from agents import *
from agents.evaluations import rollout_evaluation
from dealers import pacman_dealer

"""
Policy switching carries with it the theoretical guarantee that at each time step, it will perform at least as well as 
the best policy in its set. However, a certain number of pulls is required to accurately estimate the value of a given 
policy. If an insufficient pull budget is provided, policy switching may do somewhat worse than its best constituent
policy.
"""

if __name__ == '__main__':
    u_ro = uniform_rollout_agent.UniformRolloutAgent(depth=1, num_pulls=100)
    evaluation = rollout_evaluation.RolloutEvaluation(width=1, depth=10)
    ss = sparse_sampling_agent.SparseSamplingAgent(depth=2, pulls_per_node=5, evaluation=evaluation)
    bad_switching_agent = policy_switching_agent.PolicySwitchingAgent(depth=3, num_pulls=5, policies=[u_ro, ss])
    switching_agent = policy_switching_agent.PolicySwitchingAgent(depth=3, num_pulls=15, policies=[u_ro, ss])

    pacman = pacman_dealer.Dealer(layout_repr='testClassic')
    pacman.run(agents=[u_ro, ss, bad_switching_agent, switching_agent], num_trials=15)
