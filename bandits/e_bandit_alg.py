from abstract import absbandit_alg
import random


class EBanditAlgClass(absbandit_alg.AbstractBanditAlg):
    myname = "EGreedy Bandit Algorithm"

    def __init__(self, num_arms, epsilon=0.5):
        self.banditname = self.myname
        self.num_arms = num_arms
        self.ave_reward = [0]*num_arms
        self.num_pulls = [0]*num_arms
        self.total_pulls = 0
        self.epsilon = epsilon

    def get_bandit_name(self):
        return self.agentname

    def initialize(self):
        self.reward = [0]*self.num_arms
        self.num_pulls = [0]*self.num_arms
        self.total_pulls = 0
        self.total_pulls = 0

    def update(self, arm, reward):
        self.ave_reward[arm] = (self.ave_reward[arm] * self.num_pulls[arm] + reward)/(self.num_pulls[arm]+1)
        self.num_pulls[arm] += 1
        self.total_pulls += 1

    """
    Returns the arm with the best average reward.
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
    Returns the arm with the best average reward with 1-epsilon probability; else, returns random non-best arm.
    """
    def select_pull_arm(self):
        if self.num_arms <= 1:
            return 0
        if self.total_pulls >= self.num_arms:
            best_arm = self.select_best_arm()
            rand_val = random.random()
            if rand_val < self.epsilon:
                non_best = list(range(self.num_arms))
                non_best.remove(best_arm)
                return random.choice(non_best)
            else:
                return best_arm
        else:
            return self.total_pulls

    def get_best_reward(self):
        best_arm = self.select_best_arm()
        return self.ave_reward[best_arm]

    def get_num_pulls(self, arm):
        return self.num_pulls[arm]
