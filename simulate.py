import dealer


def run(initial_state, agents_list, simulation_count=10):
    """Simulate the given state using the provided agents the given number of times.

    Side-effect: writes output to output.txt.
    Compatible with: Connect4, Othello, Tetris, Tic-tac-toe, and Yahtzee

    :param initial_state: a game simulator structure initialized to a game's starting state.
    :param agents_list: a list of agents.
    :param simulation_count: how many games should be run.
    """
    players_count = agents_list.__len__()
    output_file = open("output.txt", "w")
    output_file.write("PLAYING " + "\n")
    output_file.write("TOTAL SIMULATIONS : " + str(simulation_count) + "\n")

    print("-" * 50)
    print("\n\nPLAYING " + "\n")
    print("TOTAL SIMULATIONS : ", simulation_count)

    dealer_object = dealer.DealerClass(agents_list, initial_state, num_simulations=simulation_count, sim_horizon=50,
                                       results_file=output_file, verbose=True)

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
