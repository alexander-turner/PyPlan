from agents import *
from agents.evaluations import rollout_evaluation
from dealers import native_dealer, pacman_dealer

"""
Compare the performance of look-ahead algorithms with different parameters.
"""

if __name__ == '__main__':
    rollout = rollout_evaluation.RolloutEvaluation(width=1, depth=5)

    ss_d3 = sparse_sampling_agent.SparseSamplingAgent(depth=3, pulls_per_node=20)
    ss_d3_rollout = sparse_sampling_agent.SparseSamplingAgent(depth=3, pulls_per_node=20, evaluation=rollout)
    ss_d5 = sparse_sampling_agent.SparseSamplingAgent(depth=5, pulls_per_node=20)
    ss_d5_rollout = sparse_sampling_agent.SparseSamplingAgent(depth=5, pulls_per_node=20, evaluation=rollout)

    uct = uct_agent.UCTAgent(depth=10, max_width=1, num_trials=1000, c=1)
    e_root_uct = e_root_uct_agent.ERootUCTAgent(depth=10, max_width=1, num_trials=1000, c=1)

    fsss = fsss_agent.FSSSAgent(depth=3, pulls_per_node=10, num_trials=1000)

    dealer = native_dealer.Dealer()
    #dealer.run(agents=[ss_d3, ss_d5], num_trials=9, env_name='Connect4')
    #dealer.run(agents=[ss_d5, uct], num_trials=9, env_name='Connect4')
    #dealer.run(agents=[uct, e_root_uct], num_trials=9, env_name='Connect4')
    #dealer.run(agents=[uct, fsss], num_trials=9, env_name='Connect4')

    dealer = pacman_dealer.Dealer(layout_repr='testClassic')
    dealer.run(agents=[ss_d3, ss_d5], num_trials=9)
