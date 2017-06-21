from agents import rollout_agent
from bandits import uniform_bandit_alg

class UniformRolloutAgentClass(rollout_agent.RolloutAgentClass):
    myname = "Uniform Rollout Agent"

    def __init__(self, depth, num_pulls, policy):
        rollout_agent.RolloutAgentClass.__init__(self, depth = depth, num_pulls = num_pulls,
                                                 policy = policy,
                                                 BanditClass = uniform_bandit_alg.UniformBanditAlgClass)
        self.agentname = self.myname