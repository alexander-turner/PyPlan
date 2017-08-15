from agents import rollout_agent
from bandits import e_bandit


class ERolloutAgent(rollout_agent.RolloutAgent):
    name = "Epsilon-Greedy Rollout Agent"

    def __init__(self, depth, num_pulls, epsilon=0.5, policy=None):
        rollout_agent.RolloutAgent.__init__(self, depth=depth, num_pulls=num_pulls,
                                            policy=policy,
                                            bandit_class=e_bandit.EBandit,
                                            bandit_parameters=epsilon)

        if policy is not None:  # if policy isn't random, it's a nested agent
            self.name = "Nested " + self.name
        self.name += " (d={}, n={}, e={})".format(depth, num_pulls, epsilon)

