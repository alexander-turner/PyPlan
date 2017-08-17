import multiprocessing
import progressbar
import random
import statistics
import time
import tabulate
from abstract import abstract_dealer
from dealers.simulators import *


class Dealer(abstract_dealer.AbstractDealer):
    """
    Facilitates simulation of Chess, Connect-4, Othello, Tetris, Tic-Tac-Toe, and Yahtzee.

    Generally, if the simulator isn't using any external frameworks to run, it should interface with this class,
    rather than its own (à la openai_sim.py's implementation).
    """
    simulators = {'Chess': chess.ChessState(), 'Connect4': connect4.Connect4State(),
                  'Othello': othello.OthelloState(), 'Tetris': tetris.TetrisState(),
                  'Tictactoe': tictactoe.TicTacToeState(), 'Yahtzee': yahtzee.YahtzeeState()}

    def __init__(self, simulation_horizon=500):
        self.simulation_horizon = simulation_horizon

        self.simulation_history = []  # the moves taken in each simulation
        self.game_winner_list = []  # a list of which agent index won which game
        self.avg_move_time = []  # average time taken by each player per move

        self.env_name = ''
        self.agents = []
        self.player_count = 0
        self.num_trials = 0
        self.show_moves = False

    def reinitialize(self):
        """Resets tracking stats for another use."""
        self.simulation_history = []
        self.game_winner_list = []
        self.avg_move_time = []  # time taken per move by each player over all games

    def configure(self, agents, num_trials, simulator_str=None, show_moves=None):
        """Reconfigure the dealer with the given parameters."""
        self.agents = agents
        self.player_count = len(agents)
        self.num_trials = num_trials
        if simulator_str:
            self.env_name = simulator_str
        if show_moves:
            self.show_moves = show_moves

    def run(self, agents, num_trials, env_name, output_path=None, multiprocess_mode='trials', show_moves=True):
        """Simulate the given state using the provided agents the given number of times.

        Compatible with: Connect4, Othello, Tetris, Tic-tac-toe, and Yahtzee.

        :param agents: a list of agents. Currently does not support iterating over multiple groups.
        :param num_trials: how many games should be run.
        :param env_name: a string indicating which simulator to use.
        :param output_path: a text file to which results should be written.
        :param multiprocess_mode: 'trials' for trial-wise multiprocessing, 'bandit' to multiprocess bandit arm pulls.
            other options will mean no multiprocessing is executed.
        :param show_moves: whether the dealer should display the game progression.
        """
        if multiprocess_mode == 'trials':
            show_moves = False  # no point in showing output if it's going to be jumbled up by multiple games at once

        self.reinitialize()
        self.configure(agents, num_trials, env_name, show_moves)

        self.run_trials(multiprocess_mode=multiprocess_mode)
        results, winner_list, avg_times = self.simulation_history, self.game_winner_list, self.avg_move_time

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
        overall_avg_reward = [avg / num_trials for avg in overall_avg_reward]

        # Tabulate win counts
        win_counts = [0] * self.player_count
        for val in winner_list:
            if val == 'draw':
                win_counts = list(map(sum, zip(win_counts, (.5, .5))))
            elif val is not None:
                win_counts[val] += 1

        # Construct the table
        table = []
        headers = ["Agent Name", "Average Final Reward", "Reward Variance", "Winrate", "Average Time / Move (s)"]
        if multiprocess_mode == 'trials':
            multiprocessing_str = "Trial-based"
        elif multiprocess_mode == 'bandit':
            multiprocessing_str = "Bandit-based"
        else:
            multiprocessing_str = "No"

        for agent_idx, agent in enumerate(agents):
            agent_rewards = [r[agent_idx] for r in overall_reward]
            table.append([agent.name,  # agent name
                          overall_avg_reward[agent_idx],  # average final reward
                          statistics.variance(agent_rewards, overall_avg_reward[agent_idx]) if self.num_trials > 1 else 0.0,
                          win_counts[agent_idx] / num_trials,  # win percentage
                          avg_times[agent_idx]])  # average time taken per move

        table = "\n" + tabulate.tabulate(table, headers, tablefmt="grid", floatfmt=".4f")
        table += "\n{} game{} of {} ran; maximum turn count was {}." \
                 " {} multiprocessing was used.\n".format(num_trials,
                                                          "s" if num_trials > 1 else "",
                                                          self.simulators[self.env_name].env_name,
                                                          self.simulation_horizon,
                                                          multiprocessing_str)
        print(table)

        if output_path is not None:
            with open(output_path, "w") as output_file:
                output_file.write(table)

    def run_trials(self, multiprocess_mode='trials'):
        bar = progressbar.ProgressBar(max_value=self.num_trials)
        game_outputs = []
        if multiprocess_mode == 'trials':
            with multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1)) as pool:
                remaining = self.num_trials
                bar.update(0)
                while remaining > 0:
                    trials_to_execute = min(pool._processes, remaining)
                    game_outputs += pool.map(self.run_trial, range(trials_to_execute))
                    remaining -= trials_to_execute
                    bar.update(self.num_trials - remaining)
                time.sleep(0.1)  # so we don't print extra progress bar
        else:
            if multiprocess_mode == 'bandit':
                old_configs = [False] * len(self.agents)
                for agent_idx, agent in enumerate(self.agents):  # enable arm-based multiprocessing
                    if hasattr(agent, 'set_multiprocess'):
                        old_configs[agent_idx] = agent.multiprocess
                        agent.set_multiprocess(True)

            for _ in bar(range(self.num_trials)):
                game_outputs.append(self.run_trial())
            time.sleep(0.1)  # so we don't print extra progress bars

            if multiprocess_mode == 'bandit':
                for agent_idx, agent in enumerate(self.agents):  # reset their multiprocess information
                    if hasattr(agent, 'set_multiprocess'):
                        agent.set_multiprocess(old_configs[agent_idx])

        for output in game_outputs:
            self.game_winner_list.append(output['winner'])
            self.simulation_history.append(output['game history'])
            self.avg_move_time += output['average move times']
        self.avg_move_time = [x / self.num_trials for x in self.avg_move_time]  # average

    def run_trial(self, sim_num=None):
        """Run a single simulation using the current configuration.

        :param sim_num: placeholder parameter that allows use with multiprocessing.Pool.
        """
        current_state = self.simulators[self.env_name].clone()
        current_state.reinitialize()

        # We want to eliminate any possible first-player advantage gained from being the first agent on the list
        if current_state.num_players > 1 and self.env_name != 'Chess':
            current_state.current_player = random.randrange(self.player_count)

        game_history, time_values = [], [[] for _ in range(current_state.num_players)]

        for h in range(self.simulation_horizon):
            if current_state.is_terminal():
                break

            # Get an action from the agent, tracking time taken
            move_start_time = time.time()
            action_to_take = self.agents[current_state.current_player].select_action(current_state)
            time_values[current_state.current_player].append(time.time() - move_start_time)

            # Take selected action
            reward = current_state.take_action(action_to_take)  # TODO take_action doesn't simulate other player

            game_history.append([reward, action_to_take])
            if self.show_moves:
                if hasattr(current_state, 'render'):  # TODO render other domains
                    current_state.render()
                else:
                    print("Agent {}".format(current_state.current_player + 1))
                    print(current_state)

        time.sleep(0.1)

        # Game statistics
        total_reward = [0.0] * self.player_count
        for turn in range(len(game_history)):
            total_reward = [x + y for x, y in zip(total_reward, game_history[turn][0])]

        if h == self.simulation_horizon:  # game is not terminal
            winner, _ = max(enumerate(total_reward), key=lambda x: x[1])  # index of highest score = player index
        else:
            winner = current_state.game_outcome

        # Calculate average time per move
        moves_per_player = float(h / self.player_count)
        time_avg = [sum(time_values[player]) / moves_per_player for player in range(current_state.num_players)]

        return {'winner': winner, 'game history': game_history, 'average move times': time_avg}

    def available_configurations(self):
        """Returns the strings for the native simulators available."""
        return list(self.simulators)
