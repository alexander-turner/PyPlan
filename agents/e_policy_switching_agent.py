from agents import switching_bandit_framework
from bandits import e_bandit


class EPolicySwitchingAgent(switching_bandit_framework.SwitchingBanditFramework):
    base_name = "0.5-Greedy Policy Switching Agent"

    def __init__(self, depth, num_pulls, policies):
        switching_bandit_framework.SwitchingBanditFramework.__init__(self, depth=depth,
                                                                     pulls_per_node=num_pulls,
                                                                     policies=policies,
                                                                     bandit_class=e_bandit.EBandit,
                                                                     bandit_parameters=0.5)

        self.name = self.base_name + " (d={}, n={}, e=0.5, policies={})".format(depth, num_pulls,
                                                                                [p.name for p in policies])
