import simulate
from simulators import *
from heuristics import *
from agents import *

rand_agent = random_agent.RandomAgentClass()

h1 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=rand_agent, width=1, depth=10)

h10 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=rand_agent, width=10, depth=10)

u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=10, num_pulls=10, policy=rand_agent)

nested_u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=5, num_pulls=10, policy=u_ro)
nested_u_ro.agentname = "Nested Uniform Rollout Agent"

e_ro_d10_n10 = e_rollout_agent.ERolloutAgentClass(depth=10, num_pulls=10, epsilon=0.5, policy=rand_agent)

ucb_ro_d100_n100_c1 = ucb_rollout_agent.UCBRolloutAgentClass(depth=100, num_pulls=100, c=1.0, policy=rand_agent)

ss_d3 = sparse_sampling_agent.SparseSamplingAgentClass(depth=3, pulls_per_node=5, heuristic=h1)
ss_d5 = sparse_sampling_agent.SparseSamplingAgentClass(depth=5, pulls_per_node=5, heuristic=h1)

uct1000 = uct_agent.UCTAgentClass(depth=10, max_width=1, num_trials=100, c=1)

eroot_uct1000 = eroot_uct_agent.ERootUCTAgentClass(depth=10, max_width=1, num_trials=1000, c=1)

policy_set = [ss_d3, uct1000]
switch_agent = policy_switch_agent.PolicySwitchAgentClass(depth=2, num_pulls=100, policies=policy_set)
e_switch_agent = e_policy_switch_agent.EPolicySwitchAgentClass(depth=2, num_pulls=100, epsilon=0.5, policies=policy_set)

openai = openai_sim.OpenAIStateClass('FrozenLake-v0', nested_u_ro, wrapper_target='Frozen_Lake', api_key='sk_brIgt2t3TLGjd0IFrWW9rw')
openai.run(100)

pacman = pacman_sim.PacmanStateClass(layout_repr='testClassic', agents=[u_ro, nested_u_ro, e_ro_d10_n10])
#pacman.run(5, verbose=True)

# TODO: investigate why e- does worse, why pacman performance is so decreased
initial_state = connect4_sim.Connect4StateClass()   # seems to be playing same game each time almost
agents_list = [switch_agent, uct1000]
#simulate.run(initial_state, agents_list)

"""
0000000
1110110
2220220
1110110
2220220
1110111
2220222

Something seems wrong with policy switching / playing against switch agent - seems like they try not to win
"""