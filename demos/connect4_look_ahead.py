from agents import *
from heuristics import rollout_heuristic
from simulators import dealer

"""
Compare the performance of look-ahead algorithms with different parameters.
"""

if __name__ == '__main__':
    h1 = rollout_heuristic.RolloutHeuristicClass(width=1, depth=10)

    ss_d3 = sparse_sampling_agent.SparseSamplingAgentClass(depth=3, pulls_per_node=5, heuristic=h1)
    ss_d5 = sparse_sampling_agent.SparseSamplingAgentClass(depth=5, pulls_per_node=5, heuristic=h1)

    uct = uct_agent.UCTAgentClass(depth=10, max_width=1, num_trials=1000, c=1)
    e_root_uct = e_root_uct_agent.ERootUCTAgentClass(depth=10, max_width=1, num_trials=1000, c=1)

    dealer = dealer.Dealer()
    dealer.run(simulator_str='connect4', agents=[ss_d3, ss_d5], num_trials=10, show_moves=False)
    dealer.run(simulator_str='connect4', agents=[ss_d5, uct], num_trials=10, show_moves=False)
    dealer.run(simulator_str='connect4', agents=[uct, e_root_uct], num_trials=10, show_moves=False)
