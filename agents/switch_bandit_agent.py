from abstract import absagent
from bandits import uniform_bandit_alg


class SwitchBanditAgentClass(absagent.AbstractAgent):
    """An agent that takes a list of policies and returns the value of the best one at a given state."""
    myname = "Policy Switching Bandit"

    def __init__(self, pulls_per_node, policies, BanditClass=None, bandit_parameters=None):
        self.agentname = self.myname
        self.num_nodes = 1
        self.pulls_per_node = pulls_per_node

        self.policies = policies

        if BanditClass is None:
            self.BanditClass = uniform_bandit_alg.UniformBanditAlgClass
        else:
            self.BanditClass = BanditClass

        self.bandit_parameters = bandit_parameters

    def get_agent_name(self):
        return self.agentname

    def select_action(self, state):
        """Selects the highest-valued action for the given state."""
        self.num_nodes = 1
        (value, action) = self.estimateV(state)
        return action

    def estimateV(self, state):
        """Returns the best expected reward and action selected by the best policy at the given state."""
        self.num_nodes += 1

        current_player = state.get_current_player()
        num_policies = len(self.policies)  # how many policies we have

        if self.bandit_parameters is None:
            bandit = self.BanditClass(num_policies)
        else:
            bandit = self.BanditClass(num_policies, self.bandit_parameters)

        # for each policy, for each player, initialize a q value
        q_values = []
        for i in range(num_policies):
            q_values.append([0]*state.number_of_players())

        for i in range(self.pulls_per_node):  # use pull budget
            policy_idx = bandit.select_pull_arm()
            policy = self.policies[policy_idx]
            [rewards, _] = policy.estimateV(state, policy.depth)
            policy.num_nodes = 1  # reset for bookkeeping purposes

            # integrate total reward with current q_values
            q_values[policy_idx] = [sum(r) for r in zip(q_values[policy_idx], rewards)]
            bandit.update(policy_idx, rewards[current_player])  # update the reward for the given arm

        # get most-selected action of highest-valued policy (useful for stochastic environments)
        best_policy_idx = bandit.select_best_arm()  # rewards
        best_action_select = self.policies[best_policy_idx].select_action(state)

        return [q / bandit.get_num_pulls(best_policy_idx) for q in q_values[best_policy_idx]], best_action_select
