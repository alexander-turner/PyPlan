from abstract import abstract_evaluation


class ZeroEvaluation(abstract_evaluation.AbstractEvaluation):
    my_name = "Zero Evaluation"

    def get_evaluation_name(self):
        return self.my_name

    def evaluate(self, state):
        """Returns zero reward for all players, regardless of state."""
        return [0]*state.number_of_players()
