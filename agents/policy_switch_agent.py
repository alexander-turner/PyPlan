from agents import switch_bandit_agent
from bandits import uniform_bandit_alg


class PolicySwitchAgentClass(switch_bandit_agent.SwitchBanditAgentClass):
    my_name = "Policy Switching Agent"

    def __init__(self, num_pulls, policies, bandit_parameters=None):
        switch_bandit_agent.SwitchBanditAgentClass.__init__(self, pulls_per_node=num_pulls,
                                                            policies=policies,
                                                            BanditClass=uniform_bandit_alg.UniformBanditAlgClass,
                                                            bandit_parameters=bandit_parameters)

        self.agent_name = self.my_name + " (n={}, policies={})".format(num_pulls, [p.agentname for p in policies])
