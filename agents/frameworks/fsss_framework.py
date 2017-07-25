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

        self.set_value_bounds(value_bounds)

    def get_agent_name(self):
        return self.agent_name

    def set_value_bounds(self, value_bounds):
        self.min_value, self.max_value = value_bounds[0], value_bounds[1]

    def select_action(self, state):
        """Selects the highest-valued action for the given state."""
        if state.is_terminal():  # there's nothing left to do
            return None

        self.num_nodes = 1

        root_node = Node(state, 0, state.get_actions())

        while True:
            self.run_trial(root_node, self.depth)
            if self.is_done(root_node):
                break

        return self.get_best_action(root_node)

    def run_trial(self, node, depth):
        """Each trial improves the accuracy of the bounds and closes one node.

        A node is closed when enough the lower and upper bounds on the state value Q(s) are equal.

        Each invocation involves the exploration of the s' with the largest state value bound discrepancy for the a with
         the greatest upper bound, and ends with the closure of one new node in the tree. The selection of a* and s*
         allows for pruning compared to Sparse Sampling. That is, if an action at the root node level looks good enough,
         we don't need to keep closing nodes in suboptimal parts of the tree; we can just make our decision.

        :param node: points to a Node object.
        :param depth: how many more layers to generate before using the heuristic.
        """
        # TODO use heaps to reduce complexity
        if node.state.is_terminal():
            node.lower[-1] = node.transition_reward
            node.upper[-1] = node.transition_reward
            return

        current_player = node.state.get_current_player()

        if depth == 0:  # reached a leaf
            state_value = self.heuristic.evaluate(node.state)
            node.lower[-1] = state_value[current_player]
            node.upper[-1] = state_value[current_player]
            return
        elif node.times_visited == 0:  # have yet to visit this node at this depth
            for action_idx in range(node.num_actions):
                node.lower[action_idx] = self.min_value
                node.upper[action_idx] = self.max_value

        best_action = self.get_best_action(node)
        best_action_idx = node.action_list.index(best_action)
        if node.action_expansions[best_action_idx] == 0:  # only generate successors for actions we explore
            for _ in range(self.pulls_per_node):  # sample C states
                sim_state = node.state.clone()  # clone so that hashing works properly
                immediate_reward = sim_state.take_action(best_action)  # simulate taking action

                if sim_state not in node.children[best_action_idx]:
                    new_node = Node(sim_state, immediate_reward[current_player], sim_state.get_actions())
                    new_node.lower[-1] = self.min_value
                    new_node.upper[-1] = self.max_value

                    node.children[best_action_idx][sim_state] = new_node

                    self.num_nodes += 1  # we've made a new Node

        child_nodes = [node.children[best_action_idx][n] for n in node.children[best_action_idx]]

        # Find the greatest difference between the upper and lower bounds for depth-1
        bound_differences = [tuple([n.state, n.upper[-1] - n.lower[-1]]) for n in child_nodes]
        successor_key = (max(bound_differences, key=lambda x: x[1]))[0]  # retrieve key from tuple
        successor_node = node.children[best_action_idx][successor_key]

        self.run_trial(successor_node, depth - 1)

        node.times_visited += 1

        # Bounds for best action in this state are the reward plus the discounted average of child bounds
        node.lower[best_action_idx] = successor_node.transition_reward + self.discount * sum([n.lower[-1]
                                                                                          for n in child_nodes]) \
                                                                     / self.pulls_per_node
        node.upper[best_action_idx] = successor_node.transition_reward + self.discount * sum([n.upper[-1]
                                                                                          for n in child_nodes]) \
                                                                     / self.pulls_per_node

        node.lower[-1] = max([node.lower[action_idx] for action_idx in range(node.num_actions)])
        node.upper[-1] = max([node.upper[action_idx] for action_idx in range(node.num_actions)])

    def is_done(self, root_node):
        """Returns whether we've found the best action at the root state.

        Specifically, this is the case when the lower bound for the best action is greater than the upper bounds of all
            non-best actions.
        """
        best_action = self.get_best_action(root_node)
        best_action_idx = root_node.action_list.index(best_action)
        for action_idx in range(root_node.num_actions):
            if action_idx == best_action_idx:
                continue
            if root_node.lower[best_action_idx] < root_node.upper[action_idx]:
                return False
        return True

    @staticmethod
    def get_best_action(node):
        """Returns the action with the maximal upper bound for the given node.state and depth."""
        upper_bounds = [tuple([action_idx, node.upper[action_idx]]) for action_idx in range(node.num_actions)]
        best_action_idx = (max(upper_bounds, key=lambda x: x[1]))[0]  # get index of best action
        return node.action_list[best_action_idx]


class Node:
    """Stores information on a state, reward for reaching the state, and the actions available"""

    def __init__(self, state, transition_reward, action_list):
        """Contains the state, reward obtained by reaching the state, actions at the state, children, and bounds."""
        self.state = state
        self.transition_reward = transition_reward
        self.action_list = action_list
        self.num_actions = len(self.action_list)

        self.num_nodes = 0
        self.times_visited = 0

        """
        Each action is associated with a dictionary that stores successor nodes.
        The key for each successor is the state.
        """
        self.children = [{} for _ in range(self.num_actions)]

        """
        The lower and upper bounds on the estimate Q^d(s, a) of the value of taking action a in state s at depth d. 
        
        The state value is stored at lower[-1] / upper[-1].
        """
        self.lower = [0] * (self.num_actions + 1)
        self.upper = [0] * (self.num_actions + 1)

        # action_expansions[action_idx] = how many times we've sampled the given action
        self.action_expansions = [0] * self.num_actions
