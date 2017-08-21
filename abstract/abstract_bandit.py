import abc
import numpy
from abc import ABCMeta


class AbstractBandit:
    """The main class for implementing bandit algorithms."""
    __metaclass__ = ABCMeta
    name = ""

    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.initialize()

    def initialize(self):
        """Reset the bandit."""
        self.average_reward = numpy.array([0.0] * self.num_arms)
        self.num_pulls = numpy.array([0] * self.num_arms)
        self.total_pulls = 0

    def update(self, arm, reward):
        """Update the arm's pull count, total pull count, and average reward (using online mean updating)."""
        self.num_pulls[arm] += 1
        self.average_reward[arm] = (self.average_reward[arm] * (self.num_pulls[arm] - 1) + reward) / self.num_pulls[arm]
        self.total_pulls += 1

    def select_best_arm(self):
        """Returns arm with the best average reward."""
        return self.average_reward.argmax()

    def get_best_reward(self):
        return self.average_reward[self.select_best_arm()]

    @abc.abstractmethod
    def select_pull_arm(self):
        """Selects and returns a pull arm pursuant to the details of the bandit algorithm."""
        raise NotImplementedError
