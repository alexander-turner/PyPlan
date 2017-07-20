from agents import recursive_bandit_framework
from heuristics import rollout_heuristic


class RolloutAgentClass(recursive_bandit_framework.RecursiveBanditAgentClass):
    my_name = "Rollout Agent"

    def __init__(self, depth, num_pulls, policy=None, bandit_class=None, bandit_parameters=None):
        h1 = rollout_heuristic.RolloutHeuristicClass(width=1, depth=depth, rollout_policy=policy)

        recursive_bandit_framework.RecursiveBanditAgentClass.__init__(self, depth=1, pulls_per_node=num_pulls,
                                                                      heuristic=h1, bandit_class=bandit_class,
                                                                      bandit_parameters=bandit_parameters)
        self.agent_name = self.my_name
