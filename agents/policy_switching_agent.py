from agents import switching_bandit_framework
from bandits import uniform_bandit


class PolicySwitchingAgent(switching_bandit_framework.SwitchingBanditFramework):
    base_name = "Policy-Switching Agent"

    def __init__(self, depth, num_pulls, policies, bandit_parameters=None):
        """Runs all of the policies using the given depth."""
        switching_bandit_framework.SwitchingBanditFramework.__init__(self, depth=depth,
                                                                     pulls_per_node=num_pulls,
                                                                     policies=policies,
                                                                     bandit_class=uniform_bandit.UniformBandit,
                                                                     bandit_parameters=bandit_parameters)

        self.name = self.base_name + " (d={}, n={}, policies={})".format(depth, num_pulls, [p.name for p in policies])
