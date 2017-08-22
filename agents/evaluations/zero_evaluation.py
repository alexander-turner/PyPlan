import numpy as np
from abstract import abstract_evaluation


class ZeroEvaluation(abstract_evaluation.AbstractEvaluation):
    name = "Zero Evaluation"
    
    def evaluate(self, state):
        """Returns zero reward for all players, regardless of state."""
        return np.array([0] * state.num_players)
