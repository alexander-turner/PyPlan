from agents import switch_bandit_framework
from bandits import uniform_bandit_alg


class PolicySwitchAgentClass(switch_bandit_framework.SwitchBanditAgentClass):
    my_name = "Policy Switching Agent"

    def __init__(self, depth, num_pulls, policies, bandit_parameters=None):
        """Runs all of the policies using the given depth."""
        switch_bandit_framework.SwitchBanditAgentClass.__init__(self, depth=depth,
                                                                pulls_per_node=num_pulls,
                                                                policies=policies,
                                                                bandit_class=uniform_bandit_alg.UniformBanditAlgClass,
                                                                bandit_parameters=bandit_parameters)

        self.agent_name = self.my_name + " (d={}, n={}, policies={})".format(depth, num_pulls,
                                                                             [p.agent_name for p in policies])
