from demos import dealer
import tabulate


def run(simulator, agents_list, num_trials=10, output_path=None, show_moves=True):
    """Simulate the given state using the provided agents the given number of times.

    Compatible with: Connect4, Othello, Tetris, Tic-tac-toe, and Yahtzee

    :param simulator: a game simulator structure initialized to a game's starting state.
    :param agents_list: a list of agents.
    :param num_trials: how many games should be run.
    :param output_path: a text file to which results should be written.
    :param show_moves: whether the dealer should display each move.
    """
    table = []
    headers = ["Agent Name", "Average Final Reward", "Winrate", "Average Time / Move (s)"]
    num_players = len(agents_list)

    simulator.initialize()  # reset the simulator state
    dealer_object = dealer.DealerClass(simulator, agents_list, num_simulations=num_trials, sim_horizon=50,
                                       verbose=show_moves)

    dealer_object.start_simulation()
    [results, winner_list, avg_times] = dealer_object.simulation_stats()

    # Calculate the results
    overall_reward = []
    for game in range(len(results)):
        game_reward_sum = [0] * num_players

        for move in range(len(results[game])):
            game_reward_sum = [x + y for x, y in zip(game_reward_sum, results[game][move][0])]

        overall_reward.append(game_reward_sum)

    overall_avg_reward = [0] * num_players
    for game in range(len(results)):
        overall_avg_reward = [x + y for x, y in zip(overall_avg_reward, overall_reward[game])]

    for x in range(len(overall_avg_reward)):
        overall_avg_reward[x] = overall_avg_reward[x] / num_trials

    # Tabulate win counts
    win_counts = [0] * num_players
    for val in winner_list:
        if val is not None:
            win_counts[val] += 1

    for agent_idx, agent in enumerate(agents_list):
        table.append([agent.agentname,  # agent name
                      overall_avg_reward[agent_idx],  # average final reward
                      win_counts[agent_idx] / num_trials,  # win percentage
                      avg_times[agent_idx]])  # average time taken per move

    table = tabulate.tabulate(table, headers, tablefmt="grid", floatfmt=".4f")
    table += "\n{} game{} of {} ran.\n\n".format(num_trials, "s" if num_trials > 1 else "", simulator.myname)
    print(table)

    if output_path is not None:
        output_file = open("output.txt", "w")
        output_file.write(table)
        output_file.close()
