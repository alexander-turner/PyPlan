from agents import *
from agents.evaluations import rollout_evaluation
from dealers import native_dealer

"""
Compare the performance of Sparse Sampling and UCT (Monte-Carlo Tree Search) as we vary parameters.
"""

if __name__ == '__main__':
    rollout = rollout_evaluation.RolloutEvaluation(width=1, depth=5)

    ss_d2 = sparse_sampling_agent.SparseSamplingAgent(depth=2, pulls_per_node=20)
    ss_d2_rollout = sparse_sampling_agent.SparseSamplingAgent(depth=2, pulls_per_node=20, evaluation=rollout)
    ss_d3 = sparse_sampling_agent.SparseSamplingAgent(depth=3, pulls_per_node=20)

    uct_t100 = uct_agent.UCTAgent(depth=10, max_width=1, num_trials=100, c=1)
    uct_t1000 = uct_agent.UCTAgent(depth=10, max_width=1, num_trials=1000, c=1)
    uct_t1000_halfc = uct_agent.UCTAgent(depth=10, max_width=1, num_trials=1000, c=0.5)

    dealer = native_dealer.Dealer()
    dealer.run(agents=[ss_d2, ss_d2_rollout], num_trials=30, env_name='Connect4')  # TODO print statements explaining
    dealer.run(agents=[ss_d2_rollout, ss_d3], num_trials=30, env_name='Connect4')
    dealer.run(agents=[ss_d2_rollout, uct_t1000], num_trials=30, env_name='Connect4')
    dealer.run(agents=[uct_t100, uct_t1000], num_trials=30, env_name='Connect4')
    dealer.run(agents=[uct_t1000, uct_t1000_halfc], num_trials=30, env_name='Connect4')
