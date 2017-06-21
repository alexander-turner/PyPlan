from agents import recursive_bandit_agent
from heuristics import rollout_heuristic


class RolloutAgentClass(recursive_bandit_agent.RecursiveBanditAgentClass):
    myname = "Rollout Agent"

    def __init__(self, depth, num_pulls, policy, BanditClass=None, bandit_parameters=None):
        h1 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=policy, width=1, depth=depth)

        recursive_bandit_agent.RecursiveBanditAgentClass.__init__(self, depth=1, pulls_per_node=num_pulls,
                                                                  heuristic=h1, BanditClass=BanditClass,
                                                                  bandit_parameters=bandit_parameters)
        self.agentname = self.myname
