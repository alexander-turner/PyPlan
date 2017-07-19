from agents import recursive_bandit_agent
from bandits import uniform_bandit_alg


class SparseSamplingAgentClass(recursive_bandit_agent.RecursiveBanditAgentClass):
    my_name = "Sparse Sampling Agent"

    def __init__(self, depth, pulls_per_node, heuristic=None):
        recursive_bandit_agent.RecursiveBanditAgentClass.__init__(self, depth=depth, pulls_per_node=pulls_per_node,
                                                                  heuristic=heuristic,
                                                                  bandit_class=uniform_bandit_alg.UniformBanditAlgClass)
        self.agent_name = self.my_name + " (d={}, pulls per node={})".format(depth, pulls_per_node)
