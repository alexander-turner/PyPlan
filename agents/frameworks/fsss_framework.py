import heapq
import math
from abstract import abstract_agent


class FSSSFramework(abstract_agent.AbstractAgent):
    """A Forward Search Sparse Sampling agent, as described by Walsh et al."""
    name = "FSSS Agent"

    def __init__(self, depth, pulls_per_node, num_trials, evaluation, discount=.5):
        self.depth = depth
        if depth < 1:
            raise Exception("Depth must be at least 1.")
        self.pulls_per_node = pulls_per_node
        self.discount = discount
        self.num_trials = num_trials
        self.evaluation = evaluation

        self.num_nodes = 1

        self.env_name = ""  # the name of the environment for which the value bounds are configured
        self.minimums = [float('-inf') for _ in range(self.depth + 1)]  # the minimum value for the given depth
        self.maximums = [float('inf') for _ in range(self.depth + 1)]

    def select_action(self, state):
        """Selects the highest-valued action for the given state."""
        if state.is_terminal():  # there's nothing left to do
            return None

        self.num_nodes = 1
        if self.env_name != state.env_name:  # if we haven't already initialized bounds for this simulator
            self.set_min_max_bounds(state)

        root_node = Node(state, 0)

        for i in range(self.num_trials):
            self.run_trial(root_node, self.depth)
            if self.is_done(root_node):
                break

        return self.get_best_action(root_node)

    def set_min_max_bounds(self, state):
        """Pre-compute all possible minimum and maximum value bounds, accounting for depth and the discount factor."""
        value_bounds = state.get_value_bounds()
        if value_bounds['evaluation function'] is not None:  # use function to evaluate each state and derive the bounds
            eval_min, eval_max = value_bounds['evaluation function'](state)
            self.minimums = [eval_min for _ in range(self.depth + 1)]
            self.minimums = [eval_max for _ in range(self.depth + 1)]

            self.evaluate_bounds = True
            return
        else:
            self.evaluate_bounds = False

        discount_powers = [pow(self.discount, k) for k in range(self.depth + 1)]
        temp_min, temp_max = 0, 0  # non-terminal minimum / maximum up to this point

        for d in range(self.depth + 1):
            # Compute minimums[d]
            if value_bounds['pre-computed min'] is not None:
                self.minimums[d] = value_bounds['pre-computed min']
            else:
                defeat = temp_min + discount_powers[d] * value_bounds['defeat']
                temp_min += discount_powers[d] * value_bounds['min non-terminal']
                to_compare = [defeat, temp_min]
                if d > 0:
                    to_compare.append(self.minimums[d-1])
                self.minimums[d] = min(to_compare)

            # Compute maximums[d]
            if value_bounds['pre-computed max'] is not None:
                self.maximums[d] = value_bounds['pre-computed max']
            else:
                victory = temp_max + discount_powers[d] * value_bounds['victory']
                temp_max += discount_powers[d] * value_bounds['max non-terminal']
                to_compare = [victory, temp_max]
                if d > 0:
                    to_compare.append(self.maximums[d-1])
                self.maximums[d] = max(to_compare)

        self.env_name = state.env_name

    def run_trial(self, node, depth):
        """Each trial improves the accuracy of the bounds and closes one node.

        A node is closed when enough the lower and upper bounds on the state value Q(s) are equal.

        Each invocation involves the exploration of the s' with the largest state value bound discrepancy for the a with
         the greatest upper bound, and ends with the closure of one new node in the tree. The selection of a* and s*
         allows for pruning compared to Sparse Sampling. That is, if an action at the root node level looks good enough,
         we don't need to keep closing nodes in suboptimal parts of the tree; we can just make our decision.

        :param node: points to a Node object.
        :param depth: how many more layers to generate before using the state evaluation function.
        """
        if node.state.is_terminal():
            node.lower_state = node.transition_reward
            node.upper_state = node.transition_reward
            return

        if self.evaluate_bounds:  # if we have a bound-evaluation function for each state
            self.set_min_max_bounds(node.state)

        current_player = node.state.get_current_player()

        if depth == 0:  # reached a leaf
            state_value = self.evaluation.evaluate(node.state)
            node.lower_state = state_value[current_player]
            node.upper_state = state_value[current_player]
            return
        elif node.times_visited == 0:
            for action_idx in range(node.num_actions):
                heapq.heappush(node.lower, (-1 * self.minimums[depth], action_idx))
                heapq.heappush(node.upper, (-1 * self.maximums[depth], action_idx))

        best_action = self.get_best_action(node)
        best_action_idx = node.action_list.index(best_action)

        if node.action_expansions[best_action_idx] < self.pulls_per_node:
            sim_state = node.state.clone()  # clone so that hashing works properly
            immediate_reward = sim_state.take_action(best_action)  # simulate taking action

            if sim_state not in node.children[best_action_idx]:
                new_node = Node(sim_state, immediate_reward[current_player])
                new_node.lower_state = self.minimums[depth-1]
                new_node.upper_state = self.maximums[depth-1]

                node.children[best_action_idx][sim_state] = new_node
                self.num_nodes += 1  # we've made a new Node
            node.children[best_action_idx][sim_state].times_sampled += 1
            node.action_expansions[best_action_idx] += 1

        child_nodes = [node.children[best_action_idx][n] for n in node.children[best_action_idx]]

        # Find the greatest difference between the upper and lower bounds for depth-1
        bound_differences = [tuple([n.state, n.upper_state - n.lower_state]) for n in child_nodes]
        successor_key, _ = max(bound_differences, key=lambda x: x[1])  # retrieve key from tuple
        successor_node = node.children[best_action_idx][successor_key]

        self.run_trial(successor_node, depth - 1)
        node.times_visited += 1

        # Bounds for best action in this state are the reward plus the discounted average of child bounds
        pulls_remaining = self.pulls_per_node - node.action_expansions[best_action_idx]

        new_lower, new_upper = self.perform_backup(child_nodes, depth, successor_node.transition_reward, pulls_remaining)

        heapq.heapreplace(node.lower, (-1 * new_lower, best_action_idx))  # pop the old best_action_idx lower; push new
        heapq.heapreplace(node.upper, (-1 * new_upper, best_action_idx))

        node.lower_state = -1 * node.lower[0][0]  # [list_pos][value]; correct for heap inversion
        node.upper_state = -1 * node.upper[0][0]

    def perform_backup(self, child_nodes, depth, reward, pulls_remaining):
        """Perform the bound backup over the child nodes while handling nan/infinite values."""
        if math.isinf(self.minimums[depth-1]):
            if pulls_remaining > 0:
                new_lower = self.minimums[depth-1]
            else:
                lower_average = sum([n.lower_state * n.times_sampled for n in child_nodes]) / self.pulls_per_node
                new_lower = reward + self.discount * lower_average
        else:
            lower_average = (sum([n.lower_state * n.times_sampled for n in child_nodes]) +
                             self.minimums[depth-1] * pulls_remaining) \
                            / self.pulls_per_node
            new_lower = reward + self.discount * lower_average

        if math.isinf(self.maximums[depth-1]):
            if pulls_remaining > 0:
                new_upper = self.maximums[depth-1]
            else:
                upper_average = sum([n.upper_state * n.times_sampled for n in child_nodes]) / self.pulls_per_node
                new_upper = reward + self.discount * upper_average
        else:
            upper_average = (sum([n.upper_state * n.times_sampled for n in child_nodes]) +
                             self.maximums[depth-1] * pulls_remaining) \
                            / self.pulls_per_node
            new_upper = reward + self.discount * upper_average

        return new_lower, new_upper

    @staticmethod
    def is_done(root_node):
        """Returns whether we've found the best action at the root state.

        Specifically, this is the case when the lower bound for the best action is greater than the upper bounds of all
            non-best actions.
        """
        if root_node.num_actions == 1:  # if there's only one action, not much of a choice to make!
            return True
        best_lower = root_node.lower[0]  # largest (after inversion) lower bound in the heap
        best_upper = heapq.nsmallest(2, root_node.upper)  # two largest (after inversion) upper bounds
        if best_lower[1] == best_upper[0][1]:  # don't want to compare best_lower with its own upper bound
            return best_lower[0] <= best_upper[1][0]  # compare to second-best
        else:
            return best_lower[0] <= best_upper[0][0]

    @staticmethod
    def get_best_action(node):
        """Returns the action with the maximal upper bound for the given node.state and depth."""
        return node.action_list[node.upper[0][1]]  # [access largest heap element][access action index]


class Node:
    """Stores information on a state, reward for reaching the state, and the actions available"""

    def __init__(self, state, transition_reward):
        """Contains the state, reward obtained by reaching the state, actions at the state, children, and bounds."""
        self.state = state
        self.transition_reward = transition_reward
        self.action_list = state.get_actions()
        self.num_actions = len(self.action_list)

        self.num_nodes = 0
        self.times_visited = 0  # whether the node's bounds have been set
        self.times_sampled = 0  # times sampled in the pull budget - indicator of weight

        """
        Each action is associated with a dictionary that stores successor nodes.
        The key for each successor is the state.
        """
        self.children = [{} for _ in range(self.num_actions)]

        """
        The lower and upper bounds on the estimate Q^d(s, a) of the value of taking action a in state s at depth d. 
        
        Stored as tuples (-1*bound_value, action_idx) in a heap. Values are inverted for easier modification via heapq.
        """
        self.lower = []
        self.upper = []

        # Bounds on the state value
        self.lower_state = float('-inf')
        self.upper_state = float('inf')

        # action_expansions[action_idx] = how many times we've sampled the given action
        self.action_expansions = [0] * self.num_actions
