from abstract import abstract_heuristic


class ZeroHeuristicClass(abstract_heuristic.AbstractHeuristic):
    my_name = "Zero Heuristic"

    def __init__(self):
        self.heuristic_name = self.my_name

    def get_heuristic_name(self):
        return self.heuristic_name

    def evaluate(self, state):
        """Returns zero reward for all players, regardless of state."""
        return [0]*state.number_of_players()
