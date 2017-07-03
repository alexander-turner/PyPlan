from agents import recursive_bandit_agent
from heuristics import rollout_heuristic


class RolloutAgentClass(recursive_bandit_agent.RecursiveBanditAgentClass):
    my_name = "Rollout Agent"

    def __init__(self, depth, num_pulls, policy, bandit_class=None, bandit_parameters=None):
        h1 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=policy, width=1, depth=depth)

        recursive_bandit_agent.RecursiveBanditAgentClass.__init__(self, depth=1, pulls_per_node=num_pulls,
                                                                  heuristic=h1, bandit_class=bandit_class,
                                                                  bandit_parameters=bandit_parameters)
        self.agent_name = self.my_name
