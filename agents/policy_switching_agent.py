from agents import switching_bandit_framework
from bandits import uniform_bandit_alg  # TODO fix alg suffix


class PolicySwitchingAgentClass(switching_bandit_framework.SwitchingBanditAgentClass):
    my_name = "Policy Switching Agent"

    def __init__(self, depth, num_pulls, policies, bandit_parameters=None):
        """Runs all of the policies using the given depth."""
        switching_bandit_framework.SwitchingBanditAgentClass.__init__(self, depth=depth,
                                                                      pulls_per_node=num_pulls,
                                                                      policies=policies,
                                                                      bandit_class=uniform_bandit_alg.UniformBanditAlgClass,
                                                                      bandit_parameters=bandit_parameters)

        self.agent_name = self.my_name + " (d={}, n={}, policies={})".format(depth, num_pulls,
                                                                             [p.agent_name for p in policies])
