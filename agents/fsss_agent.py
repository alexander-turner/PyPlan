from bandits import *
from agents.frameworks import fsss_framework
from heuristics import rollout_heuristic


class FSSSAgentClass(fsss_framework.FSSSAgentClass):
    my_name = "Forward Search Sparse Sampling Agent"

    def __init__(self, depth, pulls_per_node, heuristic_depth=10, c=1, base_policy=None):
        heuristic = rollout_heuristic.RolloutHeuristicClass(width=1, depth=heuristic_depth, rollout_policy=base_policy)
        base_policy = heuristic.rollout_policy
        fsss_framework.FSSSAgentClass.__init__(self, depth=depth, pulls_per_node=pulls_per_node, heuristic=heuristic,
                                               bandit_alg_class=ucb_bandit_alg.UCBBanditAlgClass, bandit_parameters=c)
        # QUESTION what kind of bandit should we provide?
        self.agent_name = self.my_name + \
                          " (d={}, n={}, c={}, base policy={})".format(depth,
                                                                       pulls_per_node,
                                                                       c,
                                                                       base_policy.agent_name)
