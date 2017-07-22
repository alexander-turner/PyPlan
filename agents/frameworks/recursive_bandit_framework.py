from abstract import abstract_agent
from bandits import uniform_bandit_alg
from heuristics import zero_heuristic


class RecursiveBanditAgentClass(abstract_agent.AbstractAgent):
    """The agent blueprint."""
    my_name = "Recursive Bandit"

    def __init__(self, depth, pulls_per_node, heuristic=None, bandit_class=None, bandit_parameters=None):
        self.agent_name = self.my_name
        self.num_nodes = 1

        self.depth = depth
        self.pulls_per_node = pulls_per_node

        if heuristic is None:
            self.heuristic = zero_heuristic.ZeroHeuristicClass()
        else:
            self.heuristic = heuristic

        if bandit_class is None:
            self.bandit_class = uniform_bandit_alg.UniformBanditAlgClass
        else:
            self.bandit_class = bandit_class

        self.bandit_parameters = bandit_parameters

    def get_agent_name(self):
        return self.agent_name

    def select_action(self, state):
        """Selects the highest-valued action for the given state."""
        self.num_nodes = 1
        (value, action) = self.estimateV(state, self.depth)
        return action

    def estimateV(self, state, depth):
        """Returns the best expected reward and best action at the given state.

        :param depth: indicates how many more states for which the bandit algorithm will be run.
        """
        self.num_nodes += 1

        if depth == 0 or state.is_terminal():
            return self.heuristic.evaluate(state), None  # no more depth, so default to the heuristic

        current_player = state.get_current_player()
        action_list = state.get_actions()
        num_actions = len(action_list)

        # Create a bandit according to how many actions are available at the current state
        if self.bandit_parameters is None:
            bandit = self.bandit_class(num_actions)
        else:
            bandit = self.bandit_class(num_actions, self.bandit_parameters)

        q_values = [[0]*state.number_of_players()]*num_actions  # for each action, for each player, initialize a q value

        for _ in range(self.pulls_per_node):  # use pull budget
            arm_data = self.run_pull(state, bandit, depth)
            chosen_arm, total_reward = arm_data[0], arm_data[1]
            # Integrate total reward with current q_values
            q_values[chosen_arm] = [sum(r) for r in zip(q_values[chosen_arm], total_reward)]
            bandit.update(chosen_arm, total_reward[current_player])  # update the reward for the given arm

        best_arm_index = bandit.select_best_arm()

        return [q / bandit.get_num_pulls(best_arm_index) for q in q_values[best_arm_index]], action_list[best_arm_index]

    def run_pull(self, state, bandit, depth):
        """Choose an arm to pull, execute the action, and return the chosen arm and total reward received."""
        chosen_arm = bandit.select_pull_arm()
        current_state = state.clone()  # reset state

        immediate_reward = current_state.take_action(current_state.get_actions()[chosen_arm])
        future_reward = self.estimateV(current_state, depth - 1)[0]  # [0] references the q_values for best action
        total_reward = [sum(r) for r in zip(immediate_reward, future_reward)]

        return [chosen_arm, total_reward]
