from agents import recursive_bandit_agent
from heuristics import switching_heuristic

class PolicySwitchAgentClass(recursive_bandit_agent.RecursiveBanditAgentClass):
    myname = "Policy Switching Agent"

    def __init__(self, depth, num_pulls, policies, BanditClass = None, bandit_parameters = None):
        h1 = switching_heuristic.SwitchingHeuristicClass(switch_policies = policies, width = 1, depth = depth)

        recursive_bandit_agent.RecursiveBanditAgentClass.__init__(self, depth = 1, pulls_per_node = num_pulls,
                                                       heuristic = h1, BanditClass = BanditClass,
                                                       bandit_parameters = bandit_parameters)
        self.agentname = self.myname