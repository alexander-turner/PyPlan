import math
import numpy
from abstract import abstract_bandit


class UCBBandit(abstract_bandit.AbstractBandit):
    """Balances exploration and exploitation while minimizing time spent on sub-optimal arms.

    arm value := avg reward + exploration bonus that decreases as an arm is pulled more.
    """
    name = "UCB Bandit Algorithm"

    def __init__(self, num_arms, c=1.0):
        """Initialize the bandit with the given parameters.

        :param c: exploration constant in the UCB equation.
        """
        super().__init__(num_arms)
        self.c = c

    def select_pull_arm(self):
        """If each arm has been pulled at least once, returns arm with minimal cumulative regret.

        If there is an arm that has yet to be pulled, pull that arm.
        """
        if self.num_arms <= 1:  # return the only arm that we can
            return 0

        if self.total_pulls >= self.num_arms:  # if we've pulled each arm at least once
            ucb_array = self.average_reward + self.c * numpy.emath.sqrt(math.log(self.total_pulls) / self.num_pulls)
            return ucb_array.argmax()
        else:  # pull the first arm that has yet to be pulled
            return self.total_pulls
