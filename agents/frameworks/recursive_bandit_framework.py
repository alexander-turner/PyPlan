import multiprocessing
import numpy as np
from abstract import abstract_agent
from agents.bandits.uniform_bandit import UniformBandit
from agents.evaluations.zero_evaluation import ZeroEvaluation


class RecursiveBanditFramework(abstract_agent.AbstractAgent):
    """The agent blueprint."""
    name = "Recursive Bandit"

    def __init__(self, depth, pulls_per_node, evaluation=None, bandit_class=None, bandit_parameters=None):
        # TODO standardize part of init
        self.num_nodes = 1  # TODO remove?

        self.depth = depth
        self.pulls_per_node = pulls_per_node

        self.evaluation = evaluation if evaluation else ZeroEvaluation()

        self.bandit_class = UniformBandit if bandit_class is None else bandit_class
        self.bandit_parameters = bandit_parameters

        self.multiprocess = False

    def select_action(self, state):
        """Selects the highest-valued action for the given state."""
        self.num_nodes = 1
        return self.estimateV(state, self.depth)[1]  # return the best action

    def estimateV(self, state, depth):
        """Returns the best expected reward and best action at the given state.

        :param depth: indicates how many more states for which the bandit algorithm will be run.
        """
        self.num_nodes += 1

        if depth == 0 or state.is_terminal():
            return self.evaluation.evaluate(state), None  # no more depth, so default to the evaluation fn

        action_list = state.get_actions()
        num_actions = len(action_list)

        # Create a bandit according to how many actions are available at the current state
        bandit = self.bandit_class(num_actions) if self.bandit_parameters is None \
                 else self.bandit_class(num_actions, self.bandit_parameters)

        q_values = np.array([[0.0] * state.num_players] * num_actions)  # q-value for each action and each player
        if self.multiprocess and __name__ == '__main__':
            with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
                remaining = self.pulls_per_node
                while remaining > 0:
                    pulls_to_use = min(pool._processes, remaining)
                    outputs = pool.starmap(self.run_pull, [[state, bandit, depth]] * pulls_to_use)
                    remaining -= pulls_to_use

                    for arm_data in outputs:
                        self.update_bandit(q_values, arm_data, state.current_player, bandit)
        else:
            for _ in range(self.pulls_per_node):  # use pull budget
                arm_data = self.run_pull(state, bandit, depth)
                self.update_bandit(q_values, arm_data, state.current_player, bandit)

        best_arm_index = bandit.select_best_arm()

        return q_values[best_arm_index] / bandit.num_pulls[best_arm_index], action_list[best_arm_index]

    @staticmethod
    def update_bandit(q_values, arm_data, current_player, bandit):
        """Update the relevant arm of bandit and the q_values with the new rewards observed."""
        chosen_arm, total_reward = arm_data
        bandit.update(chosen_arm, total_reward[current_player])  # update the reward for the given arm
        q_values[chosen_arm] += total_reward

    def run_pull(self, state, bandit, depth):
        """Choose an arm to pull, execute the action, and return the chosen arm and total reward received."""
        chosen_arm = bandit.select_pull_arm()
        current_state = state.clone()  # reset state

        immediate_reward = current_state.take_action(current_state.get_actions()[chosen_arm])
        future_reward = self.estimateV(current_state, depth - 1)[0]  # [0] references the q_values for best action

        return chosen_arm, immediate_reward + future_reward
