from abstract import absheuristic

class ZeroHeuristicClass(absheuristic.AbstractHeuristic):
    myname = "Zero Heuristic"

    def __init__(self):
        self.heuristicname = self.myname

    def get_heuristic_name(self):
        return self.agentname

    def evaluate(self, state):
        return [0]*state.number_of_players()