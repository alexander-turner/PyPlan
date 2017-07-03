from abstract import absbandit_alg
import math


class UCBBanditAlgClass(absbandit_alg.AbstractBanditAlg):
    """Balances exploration and exploitation while minimizing time spent on sub-optimal arms.

    arm value := avg reward + exploration bonus that decreases as an arm is pulled more.
    """

    my_name = "UCB Bandit Algorithm"

    def __init__(self, num_arms, c=1.0):
        """Initialize the bandit with the given parameters.

        :param c: multiplier for the exploration constant in the UCB equation.
        """
        self.num_arms = num_arms
        self.c = c
        self.average_reward = [0] * num_arms
        self.num_pulls = [0]*num_arms
        self.total_pulls = 0

    def get_bandit_name(self):
        return self.my_name

    def initialize(self):
        """Reset the bandit while retaining the name, number of arms, and C value."""
        self.average_reward = [0] * self.num_arms
        self.num_pulls = [0]*self.num_arms
        self.total_pulls = 0

    def update(self, arm, reward):
        """Update the arm's pull count, total pull count, and average reward (using online mean updating)."""
        self.average_reward[arm] = (self.average_reward[arm] * self.num_pulls[arm] + reward) / (self.num_pulls[arm] + 1)
        self.num_pulls[arm] += 1
        self.total_pulls += 1

    def select_best_arm(self):
        """Returns arm with the best average reward."""
        best_arm = None
        best_average = None

        for i in range(self.num_arms):  # check the average reward of each arm
            if self.num_pulls[i] > 0:  # if we've pulled it at least once
                if best_arm is None or self.average_reward[i] > best_average:
                    best_arm = i
                    best_average = self.average_reward[i]

        return best_arm

    def select_pull_arm(self):
        """If each arm has been pulled at least once, returns arm with minimal cumulative regret.

        If there is an arm that has yet to be pulled, pull that arm.
        """
        if self.num_arms <= 1:  # return the only arm that we can
            return 0

        if self.total_pulls >= self.num_arms:  # if we've pulled each arm at least once
            best_arm = 0
            best_UCB = self.average_reward[0] + self.c * math.sqrt(math.log(self.total_pulls) / self.num_pulls[0])

            for i in range(1, self.num_arms):  # calculate cumulative regret for each arm
                UCB = self.average_reward[i] + self.c * math.sqrt(math.log(self.total_pulls) / self.num_pulls[i])
                if UCB > best_UCB:
                    best_arm = i
                    best_UCB = UCB
            return best_arm
        else:  # pull the first arm that has yet to be pulled
            return self.total_pulls

    def get_best_reward(self):
        best_arm = self.select_best_arm()
        return self.average_reward[best_arm]

    def get_num_pulls(self, arm):
        return self.num_pulls[arm]
