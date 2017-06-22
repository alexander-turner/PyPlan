from agents import recursive_bandit_agent
from bandits import e_bandit_alg
from heuristics import switching_heuristic


class EPolicySwitchAgentClass(recursive_bandit_agent.RecursiveBanditAgentClass):
    myname = "Policy Switching Agent"

    def __init__(self, depth, num_pulls, epsilon, policies):
        h1 = switching_heuristic.SwitchingHeuristicClass(switch_policies=policies, width=1, depth=depth)

        recursive_bandit_agent.RecursiveBanditAgentClass.__init__(self, depth=1, pulls_per_node=num_pulls,
                                                                  heuristic=h1, BanditClass=e_bandit_alg.EBanditAlgClass,
                                                                  bandit_parameters=epsilon)
        self.agentname = self.myname
