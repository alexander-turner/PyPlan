from agents import recursive_bandit_agent
from heuristics import rollout_heuristic
from bandits import uniform_bandit_alg

class SparseSamplingAgentClass(recursive_bandit_agent.RecursiveBanditAgentClass):
    myname = "Sparse Sampling Agent"

    def __init__(self, depth, pulls_per_node, heuristic = None):

        recursive_bandit_agent.RecursiveBanditAgentClass.__init__(self, depth = depth, pulls_per_node = pulls_per_node,
                                                       heuristic = heuristic,
                                                       BanditClass = uniform_bandit_alg.UniformBanditAlgClass)
        self.agentname = self.myname