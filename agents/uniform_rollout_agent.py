from agents import rollout_agent
from bandits import uniform_bandit


class UniformRolloutAgent(rollout_agent.RolloutAgent):
    name = "Uniform Rollout Agent"

    def __init__(self, depth, num_pulls, policy=None):
        rollout_agent.RolloutAgent.__init__(self, depth=depth, num_pulls=num_pulls,
                                            policy=policy,
                                            bandit_class=uniform_bandit.UniformBandit)
        
        if policy is not None:  # if policy isn't random, it's a nested agent
            self.name = "Nested " + self.name
        self.name += " (d={}, n={})".format(depth, num_pulls)
