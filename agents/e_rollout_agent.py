from agents import rollout_agent, random_agent
from bandits import e_bandit_alg


class ERolloutAgentClass(rollout_agent.RolloutAgentClass):
    my_name = "Epsilon-Greedy Rollout Agent"

    def __init__(self, depth, num_pulls, epsilon, policy):
        rollout_agent.RolloutAgentClass.__init__(self, depth=depth, num_pulls=num_pulls,
                                                 policy=policy,
                                                 bandit_class=e_bandit_alg.EBanditAlgClass,
                                                 bandit_parameters=epsilon)
        if not isinstance(policy, random_agent.RandomAgentClass):  # if policy isn't random, it's a nested agent
            self.my_name = "Nested " + self.my_name
        self.agent_name = self.my_name + " (d={}, n={}, e={})".format(depth, num_pulls, epsilon)

