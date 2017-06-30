from abstract import absbandit_alg


class UniformBanditAlgClass(absbandit_alg.AbstractBanditAlg):
    """Pulls each arm an approximately equal number of times (difference is at most 1)."""
    myname = "Uniform Bandit"

    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.reward = [0]*num_arms
        self.num_pulls = [0]*num_arms
        self.total_pulls = 0

    def get_bandit_name(self):
        return self.myname

    def initialize(self):
        """Reset the bandit while retaining the name and number of arms."""
        self.reward = [0]*self.num_arms
        self.num_pulls = [0]*self.num_arms
        self.total_pulls = 0

    def update(self, arm, reward):
        """Update the arm's pull count, observed total reward, and total pull count."""
        self.reward[arm] += reward
        self.num_pulls[arm] += 1
        self.total_pulls += 1

    def select_best_arm(self):
        """Returns the arm with the best average reward."""
        best_arm = None
        best_average = None

        for i in range(self.num_arms):  # check the average reward of each arm
            if self.num_pulls[i] > 0:  # if we've pulled it at least once
                if best_arm is None or (self.reward[i]/self.num_pulls[i]) > best_average:
                    best_arm = i
                    best_average = self.reward[i]/self.num_pulls[i]

        return best_arm

    def select_pull_arm(self):
        """Returns the arm that has been pulled the fewest number of times."""
        return self.num_pulls.index(min(self.num_pulls))

    def get_best_reward(self):
        best_arm = self.select_best_arm()
        return self.reward[best_arm]/self.num_pulls[best_arm]

    def get_num_pulls(self, arm):
        return self.num_pulls[arm]
