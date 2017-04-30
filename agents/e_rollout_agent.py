from agents import rollout_agent
from bandits import e_bandit_alg

class ERolloutAgentClass(rollout_agent.RolloutAgentClass):
    myname = "Epsilon-Greedy Rollout Agent"

    def __init__(self, depth, num_pulls, epsilon, policy):
        rollout_agent.RolloutAgentClass.__init__(self, depth = depth, num_pulls = num_pulls,
                                                 policy = policy,
                                                 BanditClass = e_bandit_alg.EBanditAlgClass,
                                                 bandit_parameters = epsilon)
        self.agentname = self.myname