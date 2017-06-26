from agents import rollout_agent
from bandits import ucb_bandit_alg


class UCBRolloutAgentClass(rollout_agent.RolloutAgentClass):
    myname = "UCB Rollout Agent"

    def __init__(self, depth, num_pulls, c, policy):
        rollout_agent.RolloutAgentClass.__init__(self, depth=depth, num_pulls=num_pulls,
                                                 policy=policy,
                                                 BanditClass=ucb_bandit_alg.UCBBanditAlgClass,
                                                 bandit_parameters=c)
        self.agentname = self.myname + " (d={}, n={}, c={})".format(depth, num_pulls, c)
