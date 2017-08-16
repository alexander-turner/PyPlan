from agents import rollout_agent
from bandits import ucb_bandit


class UCBRolloutAgent(rollout_agent.RolloutAgent):
    base_name = "UCB Rollout Agent"

    def __init__(self, depth, num_pulls, c, policy=None):
        rollout_agent.RolloutAgent.__init__(self, depth=depth, num_pulls=num_pulls,
                                            policy=policy,
                                            bandit_class=ucb_bandit.UCBBandit,
                                            bandit_parameters=c)

        self.name = self.base_name + " (d={}, n={}, c={})".format(depth, num_pulls, c)
