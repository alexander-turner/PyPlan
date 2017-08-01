import multiprocessing
import time
import os
import tabulate
import numpy
from abstract import abstract_dealer
from dealers.simulators import pacman_sim
import progressbar


class Dealer(abstract_dealer.AbstractDealer):
    def __init__(self, layout_representation, multiprocess_mode='trials', show_moves=False, use_graphics=True):
        """Initialize the given layout.

        :param layout_representation: either the layout filename (located in layouts/) or an actual layout object.
        :param multiprocess_mode: 'trials' for trial-wise multiprocessing, 'bandit' to multiprocess bandit arm pulls.
            other options will mean no multiprocessing is executed.
        :param show_moves: whether moves should be rendered. Disabled if multiprocess is True.
        :param use_graphics: whether to use the graphics or the text display.
        """
        self.multiprocess_mode = multiprocess_mode
        self.show_moves = show_moves
        self.use_graphics = use_graphics

        self.num_trials = 0

        self.simulator = pacman_sim.PacmanState(dealer=self, layout_representation=layout_representation)

    def run(self, agents, num_trials=1, multiprocess_mode='trials', show_moves=False):
        """Runs num_trials trials for each of the provided agents, neatly displaying results (if requested).

        :param agents: the agents whose Pacman performance will be compared.
        :param num_trials: how many times the game will be run.
        :param multiprocess_mode: 'trials' for trial-wise multiprocessing, 'bandit' to multiprocess bandit arm pulls.
            other options will mean no multiprocessing is executed.
        :param show_moves: whether moves should be rendered. Disabled if multiprocess is True.
        """
        self.show_moves = show_moves  # whether game moves should be shown
        self.num_trials = num_trials
        self.multiprocess_mode = multiprocess_mode

        table = []
        headers = ["Agent Name", "Average Final Score", "Winrate", "Average Time / Move (s)"]  # todo variance
        if multiprocess_mode == 'trials':
            multiprocessing_str = "Trial-based"
        elif multiprocess_mode == 'bandit':
            multiprocessing_str = "Bandit-based"
        else:
            multiprocessing_str = "No"

        for agent in agents:
            print('\nNow simulating: {}'.format(agent.name))
            time.sleep(0.1)
            output = self.run_trials(agent)
            table.append([agent.name,
                          numpy.mean(output['rewards']),  # average final score
                          output['wins'] / num_trials,  # win percentage
                          output['average move time']])
        time.sleep(0.1)
        print("\n" + tabulate.tabulate(table, headers, tablefmt="grid", floatfmt=".4f"))
        print("Each agent ran {} game{} of {}. {} multiprocessing was used.".format(num_trials,
                                                                                    "s" if num_trials > 1 else "",
                                                                                    self.simulator.env_name,
                                                                                    multiprocessing_str))

    def run_trials(self, agent):
        """Run a given number of games using the current configuration, recording and returning performance statistics.

        :param agent: the agent to be used in the trials.
        """
        self.simulator.set_agent(agent)

        bar = progressbar.ProgressBar(max_value=self.num_trials)
        game_outputs = []
        if self.multiprocess_mode == 'trials':
            with multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1)) as pool:
                remaining = self.num_trials
                bar.update(0)
                while remaining > 0:
                    trials_to_execute = min(pool._processes, remaining)
                    game_outputs += pool.map(self.run_trial, range(trials_to_execute))
                    remaining -= trials_to_execute
                    bar.update(self.num_trials - remaining)
                time.sleep(0.1)  # so we don't print extra progress bar
        else:  # enable arm-based multiprocessing
            if self.multiprocess_mode == 'bandit' and hasattr(agent, 'set_multiprocess'):
                old_config = agent.multiprocess
                agent.set_multiprocess(True)

            for i in bar(range(self.num_trials)):
                game_outputs.append(self.run_trial(i))
            time.sleep(0.1)  # so we don't print extra progress bars

            if self.multiprocess_mode == 'bandit' and hasattr(agent, 'set_multiprocess'):
                agent.set_multiprocess(old_config)

        rewards = []
        wins, average_move_time = 0, 0
        for output in game_outputs:
            rewards.append(output['reward'])
            wins += output['won']
            average_move_time += output['average move time']

        average_move_time /= self.num_trials

        return {'rewards': rewards, 'wins': wins, 'average move time': average_move_time}

    def run_trial(self, trial_num):
        """Using the game parameters, run and return information about the trial.

        :param trial_num: a placeholder parameter for compatibility with multiprocessing.Pool.
        """
        self.simulator.reinitialize()  # reset the game

        start_time = time.time()
        self.simulator.game.run(self.show_moves)
        time_taken = time.time() - start_time

        return {'reward': self.simulator.final_score, 'won': self.simulator.won,
                'average move time': time_taken / self.simulator.time_step_count}

    @staticmethod
    def available_configurations():
        """Returns the available level layouts."""
        configurations = []
        for filename in os.listdir("..\\dealers\\simulators\\pacmancode\\layouts"):
            configurations.append(filename[:-4])
        return configurations
