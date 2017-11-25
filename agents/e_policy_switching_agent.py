from agents import switching_bandit_framework
from agents.bandits.e_bandit import EBandit


class EPolicySwitchingAgent(switching_bandit_framework.SwitchingBanditFramework):
    base_name = "0.5-Greedy Policy Switching Agent"

    def __init__(self, depth, num_pulls, policies):
        switching_bandit_framework.SwitchingBanditFramework.__init__(self, depth=depth,
                                                                     pulls_per_node=num_pulls,
                                                                     policies=policies,
                                                                     bandit_class=EBandit,
                                                                     bandit_parameters=0.5)

        self.name = self.base_name + " (d={}, n={}, e=0.5, policies={})".format(depth, num_pulls,
                                                                                [p.name for p in policies])
