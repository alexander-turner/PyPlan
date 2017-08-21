from agents import rollout_agent
from agents.bandits import e_bandit


class ERolloutAgent(rollout_agent.RolloutAgent):
    base_name = "ε-Greedy Rollout Agent"

    def __init__(self, depth, num_pulls, epsilon=0.5, policy=None):
        rollout_agent.RolloutAgent.__init__(self, depth=depth, num_pulls=num_pulls,
                                            policy=policy,
                                            bandit_class=e_bandit.EBandit,
                                            bandit_parameters=epsilon)

        if policy:  # if policy isn't random, it's a nested agent
            self.name = "Nested " + self.base_name
        self.name = self.base_name + " (d={}, n={}, e={})".format(depth, num_pulls, epsilon)

