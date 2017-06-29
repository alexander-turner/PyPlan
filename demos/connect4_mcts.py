from agents import *
from heuristics import rollout_heuristic
from simulators import connect4_sim
from demos import simulate

"""
Observe how the performance of various look-ahead algorithms changes as parameters vary!
"""

if __name__ == '__main__':
    rand_agent = random_agent.RandomAgentClass()
    h1 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=rand_agent, width=1, depth=10)

    ss_d3 = sparse_sampling_agent.SparseSamplingAgentClass(depth=3, pulls_per_node=5, heuristic=h1)
    ss_d5 = sparse_sampling_agent.SparseSamplingAgentClass(depth=5, pulls_per_node=5, heuristic=h1)

    uct = uct_agent.UCTAgentClass(depth=10, max_width=1, num_trials=1000, c=1)
    e_root_uct = eroot_uct_agent.ERootUCTAgentClass(depth=10, max_width=1, num_trials=1000, c=1)

    sim = connect4_sim.Connect4StateClass()
    simulate.run(simulator=sim, agents=[ss_d3, ss_d5], num_trials=10, show_moves=False)
    simulate.run(simulator=sim, agents=[ss_d5, uct], num_trials=10, show_moves=False)
    simulate.run(simulator=sim, agents=[uct, e_root_uct], num_trials=10, show_moves=False)
