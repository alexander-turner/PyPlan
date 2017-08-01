import abc
from abc import ABCMeta


class AbstractBandit:
    """The main class for implementing bandit algorithms."""
    __metaclass__ = ABCMeta
    name = ""

    @abc.abstractmethod
    def get_bandit_name(self):
        """Returns the name of the bandit."""
        raise NotImplementedError

    @abc.abstractmethod
    def initialize(self):
        """Reset the bandit while retaining basic parameters."""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, arm, reward):
        """Update the arm's pull count, total pull count, and relevant reward structure."""
        raise NotImplementedError

    @abc.abstractmethod
    def select_best_arm(self):
        """Returns the arm with the best average reward."""
        raise NotImplementedError

    @abc.abstractmethod
    def select_pull_arm(self):
        """Selects and returns a pull arm pursuant to the details of the bandit algorithm."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_best_reward(self):
        """Returns the reward of the best arm."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_num_pulls(self, arm):
        """Returns the number of times the arm has been pulled."""
        raise NotImplementedError
