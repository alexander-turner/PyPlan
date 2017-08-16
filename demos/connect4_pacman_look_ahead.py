from agents import *
from dealers import native_dealer, pacman_dealer

"""
Compare the performance of look-ahead algorithms with different parameters.
"""

if __name__ == '__main__':
    ss_d3 = sparse_sampling_agent.SparseSamplingAgent(depth=3, pulls_per_node=5)
    ss_d5 = sparse_sampling_agent.SparseSamplingAgent(depth=5, pulls_per_node=5)

    uct = uct_agent.UCTAgent(depth=10, max_width=1, num_trials=1000, c=1)
    e_root_uct = e_root_uct_agent.ERootUCTAgent(depth=10, max_width=1, num_trials=1000, c=1)

    fsss = fsss_agent.FSSSAgent(depth=3, pulls_per_node=10, num_trials=1000)

    dealer = native_dealer.Dealer()
    dealer.run(agents=[ss_d3, ss_d5], num_trials=10, env_name='connect4')
    dealer.run(agents=[ss_d5, uct], num_trials=10, env_name='connect4')
    dealer.run(agents=[uct, e_root_uct], num_trials=10, env_name='connect4')
    dealer.run(agents=[uct, fsss], num_trials=10, env_name='connect4')

    dealer = pacman_dealer.Dealer(layout_repr='testClassic')
    dealer.run(agents=[ss_d3, ss_d5, uct, e_root_uct, fsss], num_trials=10)
