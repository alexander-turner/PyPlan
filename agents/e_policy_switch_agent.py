from agents import switch_bandit_agent
from bandits import e_bandit_alg
from heuristics import rollout_heuristic


class EPolicySwitchAgentClass(switch_bandit_agent.SwitchBanditAgentClass):
    my_name = "e-Greedy Policy Switching Agent"

    def __init__(self, num_pulls, epsilon, policies):
        switch_bandit_agent.SwitchBanditAgentClass.__init__(self, pulls_per_node=num_pulls,
                                                            policies=policies,
                                                            bandit_class=e_bandit_alg.EBanditAlgClass,
                                                            bandit_parameters=epsilon)

        self.agent_name = self.my_name + " (n={}, e={}, policies={})".format(num_pulls, epsilon,
                                                                             [p.agent_name for p in policies])
