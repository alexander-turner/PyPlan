import random
from abstract import abstract_agent
from bandits import uniform_bandit_alg
from heuristics import zero_heuristic


class MCTSAgentClass(abstract_agent.AbstractAgent):
    my_name = "MCTS Agent"

    def __init__(self, depth, max_width, num_trials, heuristic=None, bandit_alg_class=None, bandit_parameters=None,
                 root_bandit_alg_class=None, root_bandit_parameters=None):
        self.agent_name = self.my_name
        self.num_nodes = 1

        self.depth = depth
        self.max_width = max_width
        self.num_trials = num_trials

        if heuristic is None:
            self.heuristic = zero_heuristic.ZeroHeuristicClass()
        else:
            self.heuristic = heuristic

        self.bandit_parameters = bandit_parameters

        if bandit_alg_class is None:
            self.BanditAlgClass = uniform_bandit_alg.UniformBanditAlgClass
        else:
            self.BanditAlgClass = bandit_alg_class

        if root_bandit_alg_class is None:
            self.RootBanditAlgClass = self.BanditAlgClass
        else:
            self.RootBanditAlgClass = root_bandit_alg_class

        self.root_bandit_parameters = root_bandit_parameters

    def get_agent_name(self):
        return self.agent_name

    def select_action(self, state):
        """Selects the highest-valued action for the given state."""
        if self.depth == 0 or state.is_terminal():  # there's nothing left to do
            return None

        self.num_nodes = 1

        action_list = state.get_actions()

        # create a bandit according to how many actions are available at the current state
        if self.root_bandit_parameters is None:
            bandit = self.RootBanditAlgClass(len(action_list))
        else:
            bandit = self.RootBanditAlgClass(len(action_list), self.root_bandit_parameters)

        root_node = BanditNode(state, 0, action_list, bandit)

        for i in range(self.num_trials):
            self.run_trial(root_node, self.depth)

        return root_node.action_list[root_node.bandit.select_best_arm()]

    def run_trial(self, node, depth):
        """Using agent parameters, recurse up to depth layers from node in the constructed tree.

        :param node: points to a BanditNode object.
        :param depth: how many more layers to generate before using the heuristic.
        :return total_reward:
        """

        if node.bandit is None:  # leaf node
            return self.heuristic.evaluate(node.state)

        current_player = node.state.get_current_player()
        action_index = node.bandit.select_pull_arm()

        # if we reach max children nodes then select randomly among children
        if len(node.children[action_index]) >= self.max_width:
            # Each key is a state
            keys = list(node.children[action_index].keys())
            counts = [node.children[action_index][k][1] for k in keys]
            normalizer = sum(counts)
            counts = [c / normalizer for c in counts]  # list of counts proportional to total number of samples

            successor_index = multinomial(counts)  # randomly sample from polynomial counts - greater counts more likely
            successor_node = node.children[action_index][keys[successor_index]][0]

            immediate_reward = successor_node.transition_reward
            total_reward = [x + y for (x, y) in zip(immediate_reward, self.run_trial(successor_node, depth - 1))]
        else:  # generate a new successor node
            successor_state = node.state.clone()
            # reward for taking selected action at this node
            immediate_reward = successor_state.take_action(node.action_list[action_index])
            successor_actions = successor_state.get_actions()

            # if the successor state is already a child
            if successor_state in node.children[action_index]:
                successor_node = node.children[action_index][successor_state][0]
                # increment how many times successor_state has been sampled
                node.children[action_index][successor_node.state][1] += 1
                # recurse downwards into the constructed tree
                total_reward = [x + y for (x, y) in zip(immediate_reward, self.run_trial(successor_node, depth - 1))]
            else:
                if successor_state.is_terminal() or depth == 1:  # indicate it's time to use the heuristic
                    successor_bandit = None
                elif self.bandit_parameters is None:  # create a bandit according to how many actions are available
                    successor_bandit = self.BanditAlgClass(len(successor_actions))
                else:
                    successor_bandit = self.BanditAlgClass(len(successor_actions), self.bandit_parameters)

                successor_node = BanditNode(successor_state, immediate_reward, successor_actions, successor_bandit)
                node.children[action_index][successor_node.state] = [successor_node, 1]
                self.num_nodes += 1  # we've made a new BanditNode

                total_reward = [x + y for (x, y) in zip(immediate_reward, self.heuristic.evaluate(successor_state))]

        node.bandit.update(action_index, total_reward[current_player])
        return total_reward


class BanditNode:
    """Stores information on a state, reward for reaching the state, actions available, and the bandit to be used."""
    def __init__(self, state, transition_reward, action_list, bandit):
        self.state = state
        self.transition_reward = transition_reward
        self.action_list = action_list
        self.bandit = bandit

        self.num_nodes = 0
        #self.hash = self.state.__hash__()

        """
        Each action is associated with a dictionary that stores successor bandits/states.
        The key for each successor is the state. 
        The value is a list [n,c], where n is a BanditNode and c is the number of times that n has been sampled.
        """
        self.children = [{} for _ in range(len(action_list))]


def multinomial(p):
    """Samples randomly from the multinomial p."""
    r = random.random()
    val = 0
    for i in range(len(p)):
        val += p[i]
        if r <= val:
            return i
