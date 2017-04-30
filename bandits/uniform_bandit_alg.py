from abstract import absbandit_alg

class UniformBanditAlgClass(absbandit_alg.AbstractBanditAlg):
    myname = "Uniform Bandit"

    def __init__(self, num_arms):
        self.banditname = self.myname
        self.num_arms = num_arms
        self.reward = [0]*num_arms
        self.num_pulls = [0]*num_arms
        self.total_pulls = 0

    def get_bandit_name(self):
        return self.agentname

    def initialize(self):
        self.reward = [0]*num_arms
        self.num_pulls = [0]*num_arms
        self.total_pulls = 0

    def update(self,arm,reward):
        self.reward[arm] += reward
        self.num_pulls[arm] += 1
        self.total_pulls += 1

    def select_best_arm(self):
        best_ave = None
        best_arm = None

        for i in range(self.num_arms):
            if self.num_pulls[i] > 0:
                if best_arm == None:
                    best_arm = i
                    best_ave = self.reward[i]/self.num_pulls[i]
                elif (self.reward[i]/self.num_pulls[i]) > best_ave:
                    best_arm = i
                    best_ave = self.reward[i]/self.num_pulls[i]

        return best_arm

    def select_pull_arm(self):
        return self.num_pulls.index(min(self.num_pulls))

    def get_best_reward(self):
        best_arm = self.select_best_arm()
        return reward[best_arm]/num_pulls[best_arm]

    def get_num_pulls(self, arm):
        return self.num_pulls[arm]
