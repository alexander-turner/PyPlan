from abstract import absbandit_alg
import math


class UCBBanditAlgClass(absbandit_alg.AbstractBanditAlg):
    myname = "UCB Bandit Algorithm"

    def __init__(self, num_arms, C=1.0):
        self.banditname = self.myname
        self.num_arms = num_arms
        self.ave_reward = [0]*num_arms
        self.num_pulls = [0]*num_arms
        self.total_pulls = 0
        self.C = C

    def get_bandit_name(self):
        return self.agentname

    def initialize(self):
        self.reward = [0]*self.num_arms
        self.num_pulls = [0]*self.num_arms
        self.total_pulls = 0
        self.total_pulls = 0

    """
    Update the arm's pull count, total pull count, and average reward (using online mean updating)
    """
    def update(self, arm, reward):
        self.ave_reward[arm] = (self.ave_reward[arm] * self.num_pulls[arm] + reward)/(self.num_pulls[arm]+1)
        self.num_pulls[arm] += 1
        self.total_pulls += 1

    """
    Returns arm with the best average reward.
    """
    def select_best_arm(self):
        best_ave = None
        best_arm = None

        for i in range(self.num_arms):
            if self.num_pulls[i] > 0:
                if best_arm is None:
                    best_arm = i
                    best_ave = self.ave_reward[i]
                elif self.ave_reward[i] > best_ave:
                    best_arm = i
                    best_ave = self.ave_reward[i]

        return best_arm

    """
    If each arm has been pulled at least once, returns arm with minimal cumulative regret:
        avg reward + exploration bonus that decreases as an arm is pulled more.
    If there is an arm that has yet to be pulled, pull that arm.
    """
    def select_pull_arm(self):
        if self.num_arms <= 1:
            return 0

        if self.total_pulls >= self.num_arms:
            best_arm = 0
            best_UCB = self.ave_reward[0] + self.C * math.sqrt(math.log(self.total_pulls) / self.num_pulls[0])

            for i in range(1, self.num_arms):
                UCB = self.ave_reward[i] + self.C * math.sqrt(math.log(self.total_pulls) / self.num_pulls[i])
                if best_UCB < UCB:
                    best_arm = i
                    best_UCB = UCB
            return best_arm
        else:
            return self.total_pulls

    def get_best_reward(self):
        best_arm = self.select_best_arm()
        return self.ave_reward[best_arm]

    def get_num_pulls(self, arm):
        return self.num_pulls[arm]
