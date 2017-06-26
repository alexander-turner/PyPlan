from agents import switch_bandit_agent
from agents import recursive_bandit_agent
from bandits import uniform_bandit_alg
from heuristics import switching_heuristic
from heuristics import rollout_heuristic


class PolicySwitchAgentClass(switch_bandit_agent.SwitchBanditAgentClass):
    myname = "Policy Switching Agent"

    def __init__(self, depth, num_pulls, policies, bandit_parameters=None):
        # make one heuristic for each policy
        heuristics = [rollout_heuristic.RolloutHeuristicClass(rollout_policy=p, width=1, depth=depth) for p in policies]

        switch_bandit_agent.SwitchBanditAgentClass.__init__(self, pulls_per_node=num_pulls,
                                                            heuristics=heuristics,
                                                            BanditClass=uniform_bandit_alg.UniformBanditAlgClass,
                                                            bandit_parameters=bandit_parameters)

        self.agentname = self.myname
