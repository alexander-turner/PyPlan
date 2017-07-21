from abstract import abstract_dealer
from dealers.simulators import pacman_sim
import multiprocessing
import time
import os
import tabulate
import numpy


class Dealer(abstract_dealer.AbstractDealer):
    def __init__(self, layout_representation, multiprocess=True, show_moves=False, use_graphics=True):
        """Initialize the given layout.

        :param layout_representation: either the layout filename (located in layouts/) or an actual layout object.
        :param multiprocess: whether to speed the computation with parallel processing.
        :param show_moves: whether moves should be rendered. Disabled if multiprocess is True.
        :param use_graphics: whether to use the graphics or the text display.
        """
        self.multiprocess = multiprocess
        self.show_moves = show_moves
        self.use_graphics = use_graphics

        self.num_trials = 0

        self.simulator = pacman_sim.PacmanState(dealer=self, layout_representation=layout_representation)

    def run(self, agents, num_trials=1, multiprocess=True, show_moves=False):
        """Runs num_trials trials for each of the provided agents, neatly displaying results (if requested).

        :param agents: the agents whose Pacman performance will be compared.
        :param num_trials: how many times the game will be run.
        :param multiprocess: whether to speed the computation with parallel processing.
        :param show_moves: whether moves should be rendered. Disabled if multiprocess is True.
        """
        self.show_moves = show_moves  # whether game moves should be shown
        self.num_trials = num_trials
        self.multiprocess = multiprocess

        table = []
        headers = ["Agent Name", "Average Final Score", "Winrate", "Average Time / Move (s)"]

        for agent in agents:
            print('\nNow simulating: {}'.format(agent.agent_name))
            output = self.run_trials(agent)
            table.append([agent.agent_name,
                          numpy.mean(output['rewards']),  # average final score
                          output['wins'] / num_trials,  # win percentage
                          output['average move time']])
        print("\n" + tabulate.tabulate(table, headers, tablefmt="grid", floatfmt=".4f"))
        print("Each agent ran {} game{} of {}.".format(num_trials, "s" if num_trials > 1 else "",
                                                       self.simulator.my_name))

    def run_trials(self, agent):
        """Run a given number of games using the current configuration, recording and returning performance statistics.

        :param agent: the agent to be used in the trials.
        """
        self.simulator.set_agent(agent)

        game_outputs = []
        if self.multiprocess:
            # ensures the system runs smoothly
            pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1))
            game_outputs = pool.map(self.run_trial, range(self.num_trials))
            pool.close()
        else:
            for i in range(self.num_trials):
                game_outputs.append(self.run_trial(i))

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
