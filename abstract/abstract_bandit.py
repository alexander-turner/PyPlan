import abc
import operator
from abc import ABCMeta


class AbstractBandit:
    """The main class for implementing bandit algorithms."""
    __metaclass__ = ABCMeta
    name = ""

    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.average_reward = [0] * num_arms
        self.num_pulls = [0] * num_arms
        self.total_pulls = 0

    def initialize(self):
        """Reset the bandit while retaining the name, number of arms, and C value."""
        self.average_reward = [0] * self.num_arms
        self.num_pulls = [0] * self.num_arms
        self.total_pulls = 0

    def update(self, arm, reward):
        """Update the arm's pull count, total pull count, and average reward (using online mean updating)."""
        self.average_reward[arm] = (self.average_reward[arm] * self.num_pulls[arm] + reward) / (self.num_pulls[arm] + 1)
        self.num_pulls[arm] += 1
        self.total_pulls += 1

    def select_best_arm(self):
        """Returns arm with the best average reward."""
        best_arm, _ = max(enumerate(self.average_reward), key=operator.itemgetter(1))
        return best_arm

    def get_best_reward(self):
        best_arm = self.select_best_arm()
        return self.average_reward[best_arm]

    def get_num_pulls(self, arm):
        return self.num_pulls[arm]

    def get_cumulative_regret(self):
        """Return the accrued cumulative regret."""
        # TODO make uniform
        rewards = self.rewards if hasattr(self, 'rewards') else self.average_reward  # uniform_bandit uses self.rewards
        return self.total_pulls * self.max_reward - sum(map(operator.mul, zip(rewards, self.num_pulls)))

    def get_simple_regret(self):
        return self.max_reward - self.get_best_reward()

    @abc.abstractmethod
    def select_pull_arm(self):
        """Selects and returns a pull arm pursuant to the details of the bandit algorithm."""
        raise NotImplementedError
