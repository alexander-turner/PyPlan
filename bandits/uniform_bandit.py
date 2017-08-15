from abstract import abstract_bandit


class UniformBandit(abstract_bandit.AbstractBandit):
    """Pulls each arm an approximately equal number of times (difference is at most 1)."""
    name = "Uniform Bandit"

    def select_pull_arm(self):
        """Returns the arm that has been pulled the fewest number of times."""
        return self.num_pulls.index(min(self.num_pulls))
