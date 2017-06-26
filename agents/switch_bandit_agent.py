from abstract import absagent
from bandits import uniform_bandit_alg


class SwitchBanditAgentClass(absagent.AbstractAgent):
    """An agent that takes a list of heuristics - one for each policy - and returns the best evaluation."""
    myname = "Policy Switching Bandit"

    def __init__(self, pulls_per_node, heuristics, BanditClass=None, bandit_parameters=None):
        """Heuristics is our array of policies which we are switching between."""
        self.agentname = self.myname
        self.num_nodes = 1
        self.pulls_per_node = pulls_per_node

        self.heuristics = heuristics

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
        num_heuristics = len(self.heuristics)  # how many policies we have

        if self.bandit_parameters is None:
            bandit = self.BanditClass(num_heuristics)
        else:
            bandit = self.BanditClass(num_heuristics, self.bandit_parameters)

        # for each policy, for each player, initialize a q value
        q_values = [[0]*state.number_of_players()]*num_heuristics

        for i in range(self.pulls_per_node):  # use pull budget
            chosen_policy = bandit.select_pull_arm()
            rewards = self.heuristics[chosen_policy].evaluate(state)  # (reward from simulating given policy)

            # integrate total reward with current q_values
            q_values[chosen_policy] = [sum(r) for r in zip(q_values[chosen_policy], rewards)]
            bandit.update(chosen_policy, rewards[current_player])  # update the reward for the given arm

        best_policy_index = bandit.select_best_arm()
        best_action = self.heuristics[best_policy_index].rollout_policy.select_action(state)

        return [q / bandit.get_num_pulls(best_policy_index) for q in q_values[best_policy_index]], best_action
