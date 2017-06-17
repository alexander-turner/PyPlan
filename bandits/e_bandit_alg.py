from abstract import absbandit_alg
import random


class EBanditAlgClass(absbandit_alg.AbstractBanditAlg):
    """Pulls the most rewarding arm with 1-epsilon probability; else, another arm is pulled at random.

    Compared to the uniform bandit, less time is spent on non-promising arms.
    """
    myname = "EGreedy Bandit Algorithm"

    def __init__(self, num_arms, epsilon=0.5):
        self.banditname = self.myname
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.ave_reward = [0]*num_arms
        self.num_pulls = [0]*num_arms
        self.total_pulls = 0

    def get_bandit_name(self):
        return self.agentname

    def initialize(self):
        """Reset the bandit while retaining the name, number of arms, and epsilon value."""
        self.reward = [0]*self.num_arms
        self.num_pulls = [0]*self.num_arms
        self.total_pulls = 0

    def update(self, arm, reward):
        """Update the arm's pull count, total pull count, and average reward (using online mean updating)."""
        self.ave_reward[arm] = (self.ave_reward[arm] * self.num_pulls[arm] + reward)/(self.num_pulls[arm]+1)
        self.num_pulls[arm] += 1
        self.total_pulls += 1

    def select_best_arm(self):
        """Returns the arm with the best average reward."""
        best_ave = None
        best_arm = None

        for i in range(self.num_arms):  # check the average reward of each arm
            if self.num_pulls[i] > 0:  # if we've pulled it at least once
                if best_arm is None:
                    best_arm = i
                    best_ave = self.ave_reward[i]
                elif self.ave_reward[i] > best_ave:
                    best_arm = i
                    best_ave = self.ave_reward[i]

        return best_arm

    def select_pull_arm(self):
        """Returns the arm with the best average reward with 1-epsilon probability; else, returns random non-best arm.
        """
        if self.num_arms <= 1:  # return the only arm that we can
            return 0

        if self.total_pulls >= self.num_arms:  # if we've pulled each arm at least once
            best_arm = self.select_best_arm()
            rand_val = random.random()
            if rand_val < self.epsilon:
                non_best = list(range(self.num_arms))
                non_best.remove(best_arm)
                return random.choice(non_best)  # pull a random non-best arm with 1-epsilon probability
            else:  # pull the best arm with 1-epsilon probability
                return best_arm
        else:  # pull the first arm that has yet to be pulled
            return self.total_pulls

    def get_best_reward(self):
        best_arm = self.select_best_arm()
        return self.ave_reward[best_arm]

    def get_num_pulls(self, arm):
        return self.num_pulls[arm]
