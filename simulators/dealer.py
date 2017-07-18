import random
import timeit
import tabulate
import multiprocessing


class Dealer:
    """Facilitates simulation of Connect-4, Othello, Tetris, Tic-Tac-Toe, and Yahtzee."""
    def __init__(self, sim_horizon=50):
        self.simulation_horizon = sim_horizon

        self.simulation_history = []  # the moves taken in each simulation
        self.game_winner_list = []  # a list of which agent index won which game
        self.avg_move_time = []  # average time taken by each player per move

        self.simulator = None
        self.agents = []
        self.player_count = 0
        self.num_trials = 0
        self.show_moves = False

    def reinitialize(self):
        """Resets tracking stats for another use."""
        self.simulation_history = []
        self.game_winner_list = []
        self.avg_move_time = []  # time taken per move by each player over all games

    def configure(self, agents, num_trials, simulator=None, show_moves=None):
        """Reconfigure the dealer with the given parameters."""
        self.agents = agents
        self.player_count = len(agents)
        self.num_trials = num_trials
        if simulator:
            self.simulator = simulator
            self.simulator.reinitialize()
        if show_moves:
            self.show_moves = show_moves

    def run(self, simulator, agents, num_trials=10, output_path=None, multiprocess=True, show_moves=True):
        """Simulate the given state using the provided agents the given number of times.

        Compatible with: Connect4, Othello, Tetris, Tic-tac-toe, and Yahtzee.

        :param simulator: a game simulator structure initialized to a game's starting state.
        :param agents: a list of agents. Currently does not support iterating over multiple groups.
        :param num_trials: how many games should be run.
        :param output_path: a text file to which results should be written.
        :param multiprocess: whether to use parallel processing to speed simulations.
            If enabled, show_moves will be disabled.
        :param show_moves: whether the dealer should display the game progression.
        """
        if multiprocess:
            show_moves = False  # no point in showing output if it's going to be jumbled up by multiple games at once

        self.reinitialize()
        self.configure(agents, num_trials, simulator, show_moves)

        self.run_trials(multiprocess=multiprocess)
        [results, winner_list, avg_times] = self.simulation_stats()

        # Calculate the results
        overall_reward = []
        for game in range(len(results)):
            game_reward_sum = [0] * self.player_count

            for move in range(len(results[game])):
                game_reward_sum = [x + y for x, y in zip(game_reward_sum, results[game][move][0])]

            overall_reward.append(game_reward_sum)

        overall_avg_reward = [0] * self.player_count
        for game in range(len(results)):
            overall_avg_reward = [x + y for x, y in zip(overall_avg_reward, overall_reward[game])]

        for x in range(len(overall_avg_reward)):
            overall_avg_reward[x] = overall_avg_reward[x] / num_trials

        # Tabulate win counts
        win_counts = [0] * self.player_count
        for val in winner_list:
            if val is not None:
                win_counts[val] += 1

        # Construct the table
        table = []
        headers = ["Agent Name", "Average Final Reward", "Winrate", "Average Time / Move (s)"]
        for agent_idx, agent in enumerate(agents):
            table.append([agent.agent_name,  # agent name
                          overall_avg_reward[agent_idx],  # average final reward
                          win_counts[agent_idx] / num_trials,  # win percentage
                          avg_times[agent_idx]])  # average time taken per move

        table = "\n" + tabulate.tabulate(table, headers, tablefmt="grid", floatfmt=".4f")
        table += "\n{} game{} of {} ran.\n\n".format(num_trials, "s" if num_trials > 1 else "", simulator.my_name)
        print(table)

        if output_path is not None:
            with open(output_path, "w") as output_file:
                output_file.write(table)

    def run_trials(self, multiprocess=True):
        game_outputs = []
        if multiprocess:
            # ensures that the system still runs smoothly
            pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1))
            game_outputs = pool.map(self.run_trial, range(self.num_trials))
        else:
            for sim_num in range(self.num_trials):
                game_outputs.append(self.run_trial())
        for output in game_outputs:
            self.game_winner_list.append(output['winner'])
            self.write_simulation_history(output['game history'])
            self.avg_move_time += output['average move times']
        self.avg_move_time = [x / self.num_trials for x in self.avg_move_time]  # average

    def run_trial(self, sim_num=None):
        """Run a single simulation using the current configuration.

        :param sim_num: placeholder parameter that allows use with multiprocessing.Pool.
        """
        current_state = self.simulator.clone()
        current_state.reinitialize()

        # We want to eliminate any possible first-player advantage gained from being the first agent on the list
        current_state.set_current_player(random.randrange(self.player_count))

        game_history = []
        time_values = []
        h = 0

        while current_state.is_terminal() is False and h < self.simulation_horizon:
            current_player = current_state.get_current_player()

            # Get an action from the agent
            move_start_time = timeit.default_timer()
            action_to_take = self.agents[current_player].select_action(current_state)
            move_end_time = timeit.default_timer()

            if self.show_moves:
                print(current_state)
                print("Agent {}".format(current_player + 1))
                print("Time for last move: {}".format(move_end_time - move_start_time))

            # Track time taken
            time_values.append([current_player, move_end_time - move_start_time])

            # Take selected action
            reward = current_state.take_action(action_to_take)

            game_history.append([reward, action_to_take])
            h += 1

        if self.show_moves:
            print(current_state)

        # --------------- #
        # Game Statistics #
        # --------------- #
        total_reward = [0.0] * self.player_count
        for turn in range(len(game_history)):
            total_reward = [x + y for x, y in zip(total_reward, game_history[turn][0])]

        winner = current_state.game_outcome

        if self.show_moves:
            print("\nRewards :", total_reward)
            print("Winner :", str(winner))
            print("-" * 50)

        # Calculate average time per move
        time_sums = [0.0] * self.player_count
        moves_per_player = float(h / self.player_count)
        for val in time_values:
            time_sums[val[0]] += val[1]

        for sum_value in range(len(time_sums)):
            time_sums[sum_value] = time_sums[sum_value] / moves_per_player

        return {'winner': winner, 'game history': game_history, 'average move times': time_sums}

    def simulation_stats(self):
        return self.simulation_history, self.game_winner_list, self.avg_move_time

    def write_simulation_history(self, game_history):
        self.simulation_history.append(game_history)
