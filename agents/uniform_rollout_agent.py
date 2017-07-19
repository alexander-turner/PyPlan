from agents import rollout_agent, random_agent
from bandits import uniform_bandit_alg


class UniformRolloutAgentClass(rollout_agent.RolloutAgentClass):
    my_name = "Uniform Rollout Agent"

    def __init__(self, depth, num_pulls, policy=None):
        rollout_agent.RolloutAgentClass.__init__(self, depth=depth, num_pulls=num_pulls,
                                                 policy=policy,
                                                 bandit_class=uniform_bandit_alg.UniformBanditAlgClass)
        
        if policy is not None:  # if policy isn't random, it's a nested agent
            self.my_name = "Nested " + self.my_name
        self.agent_name = self.my_name + " (d={}, n={})".format(depth, num_pulls)
