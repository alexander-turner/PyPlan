import random
from abstract import abstract_state


class BanditProblem(abstract_state.AbstractState):
    """
    Construct a new bandit given a set of scaled binomial reward distributions (SBRDs). An SBRD is specified by a reward r
    and probability p in [0, 1]. The arm has p probability of returning r; else, it yields a reward of 0.

    For a given arm a, E[R_a] = p_a * r_a. That is, an arm's expected reward is the probability of receiving the arm's
    reward times the reward of the arm.
    """

    def __init__(self, distributions):
        """:param distributions: should be of form [[r_1, p_1], ..., [r_k, p_k]]."""
        self.SBRDs = distributions
        self.max_expected_reward = max([r_i * p_i for r_i, p_i in distributions])
        self.num_arms = len(distributions)

    def pull(self, arm):
        return self.SBRDs[arm][0] if random.random() <= self.SBRDs[arm][1] else 0
