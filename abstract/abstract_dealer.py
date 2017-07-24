import abc
from abc import ABCMeta


class AbstractDealer:
    """Dealers facilitate simulation and display results in an easy-to-read format.

    Simulators which are incompatible with native_dealer.py should have their own dealer. Compatibility with
     multiprocessing and tabulate is encouraged. openai_sim.py and pacman_sim.py contain implementation examples. If
     the dealer has access to multiple environments or game configurations / layouts, the available_configurations
     function should be provided.
    """
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def run(self, agents, num_trials=1, multiprocess_mode='trials', show_moves=False):
        """Handles data processing, initiates run_trials for each of the provided agents, and displays results.

        :param agents: the agents whose performance will be compared.
        :param num_trials: how many times the simulation will be run.
        :param multiprocess_mode: 'trials' for trial-wise multiprocessing, 'bandit' to multiprocess bandit arm pulls.
            other options will mean no multiprocessing is executed.
        :param show_moves: whether moves should be rendered. Disabled if multiprocess is True.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run_trials(self, agent):
        """Run a given number of trials using the current configuration, recording and returning performance statistics.

        :param agent: the agent to be used in the trials.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run_trial(self, trial_num):
        """Runs a single iteration of the simulator with the given agent.

        :param trial_num: a placeholder parameter for compatibility with multiprocessing.Pool.
        """
        raise NotImplementedError
