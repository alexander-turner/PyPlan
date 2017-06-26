from agents import switch_bandit_agent
from bandits import e_bandit_alg
from heuristics import rollout_heuristic


class EPolicySwitchAgentClass(switch_bandit_agent.SwitchBanditAgentClass):
    myname = "e-Greedy Policy Switching Agent"

    def __init__(self, num_pulls, epsilon, policies):
        # make heuristic for each policy
        heuristics = [rollout_heuristic.RolloutHeuristicClass(rollout_policy=p, width=1, depth=p.depth)
                      for p in policies]

        switch_bandit_agent.SwitchBanditAgentClass.__init__(self, pulls_per_node=num_pulls,
                                                            heuristics=heuristics,
                                                            BanditClass=e_bandit_alg.EBanditAlgClass,
                                                            bandit_parameters=epsilon)

        self.agentname = self.myname + " (n={}, e={}, policies=[{}])".format(num_pulls, epsilon,
                                                                             [h.rollout_policy.myname
                                                                              for h in heuristics])
