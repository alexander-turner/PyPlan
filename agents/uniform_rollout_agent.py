from agents import rollout_agent, random_agent
from bandits import uniform_bandit_alg


class UniformRolloutAgentClass(rollout_agent.RolloutAgentClass):
    myname = "Uniform Rollout Agent"

    def __init__(self, depth, num_pulls, policy):
        rollout_agent.RolloutAgentClass.__init__(self, depth=depth, num_pulls=num_pulls,
                                                 policy=policy,
                                                 BanditClass=uniform_bandit_alg.UniformBanditAlgClass)
        
        if not isinstance(policy, random_agent.RandomAgentClass):  # if policy isn't random, it's a nested agent
            self.myname = "Nested " + self.myname
        self.agentname = self.myname + " (d={}, n={})".format(depth, num_pulls)
