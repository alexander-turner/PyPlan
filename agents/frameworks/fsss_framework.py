from abstract import abstract_agent


class FSSSAgentClass(abstract_agent.AbstractAgent):
    """A Forward Search Sparse Sampling agent, as described by Walsh et al."""
    my_name = "FSSS Agent"

    def __init__(self, depth, pulls_per_node, heuristic, value_bounds=(float('-inf'), float('inf')), discount=1):
        """
        :param value_bounds: the bounds on the minimum and maximum values for the environment.
        """
        self.agent_name = self.my_name
        self.num_nodes = 1

        self.depth = depth
        if depth < 1:
            raise Exception("Depth must be at least 1.")
        self.pulls_per_node = pulls_per_node
        self.discount = discount
        self.heuristic = heuristic

        self.min_value, self.max_value = value_bounds[0], value_bounds[1]

        # Initialize the bound array-dictionary-arrays
        self.lower = [dict() for _ in range(self.depth + 1)]
        self.upper = [dict() for _ in range(self.depth + 1)]

    def get_agent_name(self):
        return self.agent_name

    def select_action(self, state):
        """Selects the highest-valued action for the given state."""
        if state.is_terminal():  # there's nothing left to do
            return None

        # Reset bookkeeping
        self.num_nodes = 1
        self.lower = [dict() for _ in range(self.depth + 1)]
        self.upper = [dict() for _ in range(self.depth + 1)]

        root_node = Node(state, 0, state.get_actions())

        while True:
            self.run_trial(root_node, self.depth)
            if self.is_done(root_node):
                break

        return self.get_best_action(root_node, self.depth)

    def run_trial(self, node, depth):
        """Derive more lower and upper bounds for the given node's state value at the given depth.

        :param node: points to a Node object.
        :param depth: how many more layers to generate before using the heuristic; 0-indexed.
        """
        # TODO use heaps to reduce complexity
        if node.state.is_terminal():
            self.lower[depth][node.state]['state_value'] = node.transition_reward
            self.upper[depth][node.state]['state_value'] = node.transition_reward
            return

        current_player = node.state.get_current_player()

        if node.times_visited == 0:
            self.lower[depth][node.state] = {}
            self.upper[depth][node.state] = {}

        if depth == 0:  # reached a leaf
            state_value = self.heuristic.evaluate(node.state)
            self.lower[depth][node.state]['state_value'] = state_value[current_player]
            self.upper[depth][node.state]['state_value'] = state_value[current_player]
            return
        elif node.times_visited == 0:  # have yet to visit this node at this depth
            for action in node.action_list:
                self.lower[depth][node.state][action] = self.min_value
                self.upper[depth][node.state][action] = self.max_value

                action_idx = node.action_list.index(action)
                for _ in range(self.pulls_per_node):  # sample C states
                    sim_state = node.state.clone()  # clone so that hashing works properly
                    immediate_reward = sim_state.take_action(action)  # simulate taking action

                    if sim_state not in node.children[action_idx]:
                        self.lower[depth - 1][sim_state] = {}
                        self.lower[depth - 1][sim_state]['state_value'] = self.min_value
                        self.upper[depth - 1][sim_state] = {}
                        self.upper[depth - 1][sim_state]['state_value'] = self.max_value

                        sim_actions = sim_state.get_actions()

                        new_node = Node(sim_state, immediate_reward[current_player], sim_actions)
                        node.children[action_idx][sim_state] = new_node
                        self.num_nodes += 1  # we've made a new Node

        best_action = self.get_best_action(node, depth)
        best_action_idx = node.action_list.index(best_action)

        # Find the greatest difference between the upper and lower bounds for depth-1
        bound_differences = [tuple([k, self.upper[depth - 1][k]['state_value'] -
                                    self.lower[depth - 1][k]['state_value']])
                             for k in node.children[best_action_idx]]
        successor_key = (max(bound_differences, key=lambda x: x[1]))[0]  # retrieve key from tuple
        successor_node = node.children[best_action_idx][successor_key]

        self.run_trial(successor_node, depth - 1)

        node.times_visited += 1

        # Bounds for best action in this state are the reward plus the discounted average of child bounds
        keys = node.children[best_action_idx].keys()
        self.lower[depth][node.state][best_action] = successor_node.transition_reward + \
                                                     self.discount * sum([self.lower[depth - 1][k]['state_value']
                                                                          for k in keys]) \
                                                     / self.pulls_per_node
        self.upper[depth][node.state][best_action] = successor_node.transition_reward + \
                                                     self.discount * sum([self.upper[depth - 1][k]['state_value']
                                                                          for k in keys]) \
                                                     / self.pulls_per_node

        self.lower[depth][node.state]['state_value'] = max([self.lower[depth][node.state][a] for a in node.action_list])
        self.upper[depth][node.state]['state_value'] = max([self.upper[depth][node.state][a] for a in node.action_list])

    def is_done(self, root_node):
        """Returns whether we've found the best action at the root state."""
        best_action = self.get_best_action(root_node, self.depth)
        for action in root_node.action_list:
            if action == best_action:
                continue
            if self.lower[self.depth][root_node.state][best_action] \
                    < self.upper[self.depth][root_node.state][action]:
                return False
        return True

    def get_best_action(self, node, depth):
        """Returns the action with the maximal upper bound for the given node.state and depth."""
        upper_bounds = [tuple([action, self.upper[depth][node.state][action]]) for action in node.action_list]
        best_action = (max(upper_bounds, key=lambda x: x[1]))[0]  # get index of best action
        return best_action


class Node:
    """Stores information on a state, reward for reaching the state, and the actions available"""

    def __init__(self, state, transition_reward, action_list):
        self.state = state
        self.transition_reward = transition_reward
        self.action_list = action_list

        self.num_nodes = 0
        self.times_visited = 0

        """
        Each action is associated with a dictionary that stores successor nodes.
        The key for each successor is the state.
        """
        self.children = [{} for _ in range(len(self.action_list))]
