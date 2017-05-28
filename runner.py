import dealer
from simulators import connect4, pacman_sim
from heuristics import rollout_heuristic, switching_heuristic
from agents import random_agent
from agents import recursive_bandit_agent
from agents import mcts_agent
from agents import rollout_agent
from agents import policy_switch_agent
from agents import uniform_rollout_agent
from agents import e_rollout_agent
from agents import ucb_rollout_agent
from agents import uct_agent
from agents import eroot_uct_agent
from agents import sparse_sampling_agent
import os

simulation_count = 10
players_count = 2

#initial_state = connect4.Connect4StateClass()

rand_agent1 = random_agent.RandomAgentClass()

rand_agent2 = random_agent.RandomAgentClass()

h1 = rollout_heuristic.RolloutHeuristicClass(rollout_policy = rand_agent1, width = 1, depth = 10)

h10 = rollout_heuristic.RolloutHeuristicClass(rollout_policy = rand_agent1, width = 10, depth = 10)

u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth = 10, num_pulls = 50, policy = rand_agent1)

nested_u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth = 10, num_pulls = 50, policy = u_ro)

e_ro_d10_n100 = e_rollout_agent.ERolloutAgentClass(depth = 10, num_pulls = 100, epsilon = 0.5, policy = rand_agent1)

ucb_ro_d100_n100_c1 = ucb_rollout_agent.UCBRolloutAgentClass(depth = 100, num_pulls = 100, c = 1.0, policy = rand_agent1)

ss_d3_n7 = sparse_sampling_agent.SparseSamplingAgentClass(depth = 3, pulls_per_node = 7, heuristic = h1)

uct1000 = uct_agent.UCTAgentClass(depth = 10, max_width = 1, num_trials = 1000, c = 1)

eroot_uct1000 = eroot_uct_agent.ERootUCTAgentClass(depth = 10, max_width = 1, num_trials = 1000, c = 1)

switch_agent = policy_switch_agent.PolicySwitchAgentClass(depth = 10, num_pulls = 50, policies = [u_ro, e_ro_d10_n100])

initial_state = pacman_sim.PacmanStateClass('tinySearch', eroot_uct1000)
initial_state.game.run()

agents_list = [switch_agent, u_ro]

output_file = open("output.txt", "w")
output_file.write("PLAYING " + "\n")
output_file.write("TOTAL SIMULATIONS : " + str(simulation_count) + "\n")

print("-" * 50)
print("\n\nPLAYING " + "\n")
print("TOTAL SIMULATIONS : ", simulation_count)

dealer_object = dealer.DealerClass(agents_list, initial_state, num_simulations = simulation_count, sim_horizon=50, results_file=output_file, verbose=True)

dealer_object.start_simulation()
results = dealer_object.simulation_stats()[0]
winner_list = dealer_object.simulation_stats()[1]

# RESULTS CALCULATION
overall_reward = []
for game in range(len(results)):
    game_reward_sum = [0] * players_count

    for move in range(len(results[game])):
        game_reward_sum = [x + y for x, y in zip(game_reward_sum, results[game][move][0])]
    print("REWARD OF PLAYERS IN GAME {0} : ".format(game))

    print(game_reward_sum)
    overall_reward.append(game_reward_sum)

overall_reward_avg = [0] * players_count
for game in range(len(results)):
    overall_reward_avg = [x + y for x, y in zip(overall_reward_avg, overall_reward[game])]

for x in range(len(overall_reward_avg)):
    overall_reward_avg[x] = overall_reward_avg[x] / simulation_count

temp_print = "\nAVG OF REWARDS (FOR OVERALL SIMULATION) : " + str(overall_reward_avg)

win_counts = [0.0] * players_count

for val in range(len(winner_list)):
    if winner_list[val] is not None:
        win_counts[winner_list[val] - 1] += 1.0

for val in range(players_count):
    temp_print += "\nAVG WINS FOR AGENT {0} : {1}".format(val + 1, win_counts[val] / simulation_count)

print(temp_print)
output_file.write("\n" + temp_print + "\n")
output_file.close()
