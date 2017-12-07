import random
from abstract import abstract_bandit


class EBandit(abstract_bandit.AbstractBandit):
    """Pulls the most rewarding arm with (1 - epsilon) probability; else, another arm is pulled at random.

    Compared to the uniform bandit, less time is spent on non-promising arms.
    """
    name = "e-Greedy Bandit Algorithm"

    def __init__(self, num_arms, epsilon=0.5):
        super().__init__(num_arms)
        self.epsilon = epsilon

    def select_pull_arm(self):
        """Returns the arm with the best average reward with 1-epsilon probability; else, returns random non-best arm.
        """
        if self.num_arms <= 1:  # return the only arm that we can
            return 0

        if self.total_pulls >= self.num_arms:  # if we've pulled each arm at least once
            best_arm = self.get_best_arm()
            rand_val = random.random()
            if rand_val < self.epsilon:
                non_best = list(range(self.num_arms))
                non_best.remove(best_arm)
                return random.choice(non_best)  # pull a random non-best arm with 1-epsilon probability
            else:  # pull the best arm with 1-epsilon probability
                return best_arm
        else:  # pull the first arm that has yet to be pulled
            return self.total_pulls
