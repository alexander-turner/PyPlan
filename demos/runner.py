import simulate
from simulators import *
from heuristics import *
from agents import *

rand_agent = random_agent.RandomAgentClass()

h1 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=rand_agent, width=1, depth=10)

h10 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=rand_agent, width=10, depth=10)

u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=1, num_pulls=100, policy=rand_agent)
u_ro.agentname += " (d=1, n=100)"

nested_u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=3, num_pulls=10, policy=u_ro)
nested_u_ro.agentname = "Nested Uniform Rollout Agent (d=3, n=10)"

e_ro = e_rollout_agent.ERolloutAgentClass(depth=1, num_pulls=100, epsilon=0.5, policy=rand_agent)
e_ro.agentname += " (d=1, n=100, e=0.5)"

ucb_ro_d100_n100_c1 = ucb_rollout_agent.UCBRolloutAgentClass(depth=100, num_pulls=100, c=1.0, policy=rand_agent)

ss_d3 = sparse_sampling_agent.SparseSamplingAgentClass(depth=3, pulls_per_node=5, heuristic=h1)
ss_d3.agentname += " (d=3, n=5)"

ss_d5 = sparse_sampling_agent.SparseSamplingAgentClass(depth=5, pulls_per_node=5, heuristic=h1)

uct = uct_agent.UCTAgentClass(depth=10, max_width=1, num_trials=100, c=1)

eroot_uct1000 = eroot_uct_agent.ERootUCTAgentClass(depth=10, max_width=1, num_trials=1000, c=1)

policy_set = [u_ro, ss_d3]
switch_agent = policy_switch_agent.PolicySwitchAgentClass(num_pulls=10, policies=policy_set)
switch_agent.agentname += " ( n=10)"

e_switch_agent = e_policy_switch_agent.EPolicySwitchAgentClass(num_pulls=10, epsilon=0.5, policies=policy_set)
e_switch_agent.agentname += " (n=10, e=0.5)"

#openai = openai_sim.OpenAIStateClass('FrozenLake-v0', nested_u_ro, wrapper_target='Frozen_Lake', api_key='sk_brIgt2t3TLGjd0IFrWW9rw')
#openai.run(100)

pacman = pacman_sim.PacmanStateClass(layout_repr='testClassic', agents=[ss_d3, u_ro, switch_agent, e_switch_agent],
                                     use_graphics=False)
pacman.run(10, verbose=True)

initial_state = connect4_sim.Connect4StateClass()   # seems to be playing same game each time almost
agents_list = [switch_agent, nested_u_ro]
#simulate.run(initial_state, agents_list, simulation_count=10)
