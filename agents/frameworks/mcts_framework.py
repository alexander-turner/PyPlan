import numpy as np
import multiprocessing
import random
from abstract import abstract_agent
from agents.bandits.uniform_bandit import UniformBandit
from agents.evaluations.zero_evaluation import ZeroEvaluation


class MCTSFramework(abstract_agent.AbstractAgent):
    """A Monte-Carlo Tree Search framework."""
    name = "MCTS Agent"

    def __init__(self, depth, max_width, num_trials, evaluation=None, bandit_class=None, bandit_parameters=None,
                 root_bandit_class=None, root_bandit_parameters=None):
        self.depth = depth
        self.max_width = max_width
        self.num_trials = num_trials

        self.evaluation = evaluation if evaluation else ZeroEvaluation()

        self.bandit_class = bandit_class if bandit_class else UniformBandit
        self.bandit_parameters = bandit_parameters

        self.root_bandit_class = root_bandit_class if root_bandit_class else self.bandit_class
        self.root_bandit_parameters = root_bandit_parameters

        self.multiprocess = False

    def select_action(self, state):
        """Selects the highest-valued action for the given state."""
        if self.depth == 0 or state.is_terminal():  # there's nothing left to do
            return None

        actions = state.get_actions()

        # create a bandit according to how many actions are available at the current state
        bandit = self.root_bandit_class(len(actions), self.root_bandit_parameters) if self.root_bandit_parameters \
            else self.root_bandit_class(len(actions))

        root_node = BanditNode(state, 0, actions, bandit)

        if self.multiprocess and __name__ == '__main__':
            with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
                remaining = self.num_trials
                while remaining > 0:
                    pulls_to_use = min(pool._processes, remaining)
                    pool.starmap(self.run_trial, [[root_node, self.depth]] * pulls_to_use)
                    remaining -= pulls_to_use
        else:
            for _ in range(self.num_trials):
                self.run_trial(root_node, self.depth)

        return root_node.action_list[root_node.bandit.select_best_arm()]

    def run_trial(self, node, depth):
        """Using agent parameters, recurse up to depth layers from node in the constructed tree.

        :param node: points to a BanditNode object.
        :param depth: how many more layers to generate before using the evaluation function.
        :return total_reward: a numpy Array.
        """
        if node.bandit is None:  # leaf node
            return self.evaluation.evaluate(node.state)

        action_index = node.bandit.select_pull_arm()

        # If we reach max children nodes then select randomly among children
        if len(node.children[action_index]) >= self.max_width:
            # Each key is a state
            keys = list(node.children[action_index].keys())
            counts = np.array([node.children[action_index][k][1] for k in keys])
            counts = counts / counts.sum()  # list of counts proportional to total number of samples

            successor_index = multinomial(counts)  # randomly sample from polynomial counts - greater counts more likely
            successor_node = node.children[action_index][keys[successor_index]][0]

            total_reward = successor_node.transition_reward + self.run_trial(successor_node, depth - 1)
        else:  # generate a new successor node
            successor_state = node.state.clone()
            # Reward for taking selected action at this node
            immediate_reward = successor_state.take_action(node.action_list[action_index])

            # If the successor state is already a child
            if successor_state in node.children[action_index]:
                successor_node = node.children[action_index][successor_state][0]
                # Increment how many times successor_state has been sampled
                node.children[action_index][successor_node.state][1] += 1
                # Recurse downwards into the constructed tree
                total_reward = immediate_reward + self.run_trial(successor_node, depth - 1)
            else:
                successor_actions = successor_state.get_actions()

                if successor_state.is_terminal() or depth == 1:  # indicate it's time to use the evaluation fn
                    successor_bandit = None
                elif self.bandit_parameters is None:  # create a bandit according to how many actions are available
                    successor_bandit = self.bandit_class(len(successor_actions))
                else:
                    successor_bandit = self.bandit_class(len(successor_actions), self.bandit_parameters)

                successor_node = BanditNode(successor_state, immediate_reward, successor_actions, successor_bandit)
                node.children[action_index][successor_node.state] = [successor_node, 1]

                total_reward = immediate_reward + self.evaluation.evaluate(successor_state)

        node.bandit.update(action_index, total_reward[node.state.current_player])
        return total_reward


class BanditNode:
    """Stores information on a state, reward for reaching the state, actions available, and the bandit to be used."""
    def __init__(self, state, transition_reward, action_list, bandit):
        self.state = state
        self.transition_reward = transition_reward
        self.action_list = action_list
        self.bandit = bandit

        """
        Each action is associated with a dictionary that stores successor bandits/states.
        The key for each successor is the state. 
        The value is a list [n,c], where n is a BanditNode and c is the number of times that n has been sampled.
        """
        self.children = [{} for _ in action_list]


def multinomial(p):
    """Samples randomly from the multinomial p."""
    r = random.random()
    val = 0
    for i in range(len(p)):
        val += p[i]
        if r <= val:
            return i
