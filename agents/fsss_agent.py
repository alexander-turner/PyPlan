from agents.frameworks import fsss_framework
from heuristics import rollout_heuristic


class FSSSAgentClass(fsss_framework.FSSSAgentClass):
    """A Forward Search Sparse Sampling agent, as described by Walsh et al."""
    my_name = "Forward Search Sparse Sampling Agent"

    def __init__(self, depth, pulls_per_node, heuristic=None):
        if heuristic is None:
            heuristic = rollout_heuristic.RolloutHeuristicClass(width=1, depth=10)  # default to random rollout

        fsss_framework.FSSSAgentClass.__init__(self, depth=depth, pulls_per_node=pulls_per_node, heuristic=heuristic)

        self.agent_name = self.my_name + " (d={}, n={}, base policy={})".format(depth,
                                                                                pulls_per_node,
                                                                                heuristic.rollout_policy.agent_name)
