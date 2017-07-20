from abstract import abstract_agent
from bandits import uniform_bandit_alg
from heuristics import zero_heuristic


class FSSSAgentClass(abstract_agent.AbstractAgent):
    """A Forward Search Sparse Sampling agent, as described by Walsh et al."""
    my_name = "FSSS Agent"

    def __init__(self, depth, pulls_per_node, discount=0.9, heuristic=None,
                 bandit_alg_class=None, bandit_parameters=None,
                 root_bandit_alg_class=None, root_bandit_parameters=None):
        self.agent_name = self.my_name
        self.num_nodes = 1

        self.depth = depth
        self.pulls_per_node = pulls_per_node
        self.discount = discount  # TODO what should discount be?

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

        # Initialize the bound array-dictionary-arrays
        self.lower = [dict() for _ in range(self.depth + 1)]
        self.upper = [dict() for _ in range(self.depth + 1)]

        self.visits = [dict() for _ in range(self.depth + 1)]

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

        while True:
            self.run_trial(root_node, self.depth)
            if self.is_done(root_node.state):
                break

        return self.get_best_action(root_node.state, self.depth)

    def run_trial(self, node, depth):
        """

        :param node: points to a BanditNode object.
        :param depth: how many more layers to generate before using the heuristic.
        """
        # TODO use heaps to reduce complexity
        num_actions = len(node.action_list)
        if node.state not in self.visits[depth]:  # Question is state ok or do we need node?
            self.visits[depth][node.state] = 0
            self.lower[depth][node.state] = [0] * num_actions
            self.upper[depth][node.state] = [0] * num_actions

        if depth == 0:  # reached a leaf
            for action in range(num_actions):
                sim_state = node.state.clone()
                immediate_reward = sim_state.take_action(action)  # Question use heuristic? how do we know this is representative?
                self.lower[depth][node.state][action] = immediate_reward
                self.upper[depth][node.state][action] = immediate_reward
            self.lower[depth][node.state].value = max(self.lower[depth][node.state])
            self.upper[depth][node.state].value = max(self.upper[depth][node.state])
            return
        elif self.visits[depth][node.state] == 0:  # have yet to visit this node at this depth
            for action in range(num_actions):
                self.lower[depth][node.state][action] = float('-inf')
                self.upper[depth][node.state][action] = float('inf')
            sim_state = node.state.clone()
            for _ in range(self.pulls_per_node):  # sample C states
                action = node.bandit.select_pull_arm()
                sim_state.set(node.state)
                immediate_reward = sim_state.take_action(action)  # simulate taking action
                if sim_state not in node.children[action]:
                    self.lower[depth][sim_state].value = float('-inf')
                    self.upper[depth][sim_state].value = float('inf')

                    sim_actions = sim_state.get_actions()
                    if sim_state.is_terminal() or depth == 1:  # indicate it's time to use the heuristic
                        sim_bandit = None
                    elif self.bandit_parameters is None:  # create a bandit according to how many actions are available
                        sim_bandit = self.BanditAlgClass(len(sim_actions))
                    else:
                        sim_bandit = self.BanditAlgClass(len(sim_actions), self.bandit_parameters)

                    new_node = BanditNode(sim_state, immediate_reward, sim_actions, sim_bandit)
                    node.children[action][sim_state] = new_node
                    self.num_nodes += 1  # we've made a new BanditNode
                node.bandit.update(action, immediate_reward)  # update the bandit with the information we gained

        best_action = self.get_best_action(node.state, depth)

        # Find the greatest difference between the upper and lower bounds for depth-1
        bound_differences = [tuple([s, self.upper[depth - 1][s].value - self.lower[depth - 1][s].value, ])
                             for s in node.children[best_action]]
        successor_state = (max(bound_differences, key=lambda x: x[1]))[0]  # retrieve state from tuple
        successor_node = node.children[best_action][successor_state]

        self.run_trial(successor_node, depth - 1)

        self.visits[depth][node.state] += 1

        # Bounds for best action in this state are the reward plus the discounted average of child bounds
        self.lower[depth][node.state][best_action] = successor_node.transition_reward + \
                                                     self.discount * sum([self.lower[depth - 1][s].value
                                                                          for s in node.children[best_action]]) \
                                                     / self.pulls_per_node
        self.upper[depth][node.state][best_action] = successor_node.transition_reward + \
                                                     self.discount * sum([self.upper[depth - 1][s].value
                                                                          for s in node.children[best_action]]) \
                                                     / self.pulls_per_node

        self.lower[depth][node.state].value = max(self.lower[depth][node.state])
        self.upper[depth][node.state].value = max(self.upper[depth][node.state])

    def is_done(self, root_state):
        """Returns whether we've found the best action at the root state."""
        best_action = self.get_best_action(root_state, self.depth)
        for idx, upper_bound in enumerate(self.upper[self.depth][root_state]):
            if idx == best_action:
                continue
            if self.lower[self.depth][root_state][best_action] < upper_bound:
                return False
        return True

    def get_best_action(self, node, depth):
        """Returns the action with the maximal upper bound for the given state and depth."""
        upper_bounds = [self.upper[depth][state][action] for action in node.action_list]
        best_action, _ = max(enumerate(upper_bounds), key=lambda x: x[1])  # get index of best action
        return best_action


class BanditNode:
    """Stores information on a state, reward for reaching the state, actions available, and the bandit to be used."""

    def __init__(self, state, transition_reward, action_list, bandit):
        self.state = state
        self.transition_reward = transition_reward
        self.action_list = action_list
        self.bandit = bandit
        self.num_nodes = 0

        """
        Each action is associated with a dictionary that stores successor bandits/states.
        The key for each successor is the state. 
        The value is a list [n,c], where n is a BanditNode and c is the number of times that n has been sampled.
        """
        self.children = [{} for _ in range(len(action_list))]

