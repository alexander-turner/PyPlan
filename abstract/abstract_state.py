import abc
from abc import ABCMeta


class AbstractState:
    """The main class for implementing simulators.

    In addition to the methods below, it is important to implement __eq__ and __hash__, which are used by some planning
    methods to judge equivalence of states. If these are not implemented, then the code will work, but will perhaps
    be sub-optimal. This is because states that are fundamentally the same but correspond to distinct
    objects will be treated as non-equivalent.

    Simulators compatible with native_dealer.py must implement __str__.
    """
    __metaclass__ = ABCMeta

    num_players = 1  # the number of players in the game
    current_player = 0  # the index of the current player
    env_name = ""  # the name of the environment

    @abc.abstractmethod
    def reinitialize(self):
        """This method sets the state to the/an initial state of the domain."""
        raise NotImplementedError

    @abc.abstractmethod
    def clone(self):
        """Creates a deep copy of itself."""
        raise NotImplementedError

    @abc.abstractmethod
    def set(self, state):
        """Makes the object equivalent to state by copying the critical information."""
        raise NotImplementedError

    @abc.abstractmethod
    def take_action(self, action):
        """This method simulates the result of taking action in the state.

        It returns the resulting reward vector, where the reward is a list [r1,...,r_n] for an n agent game.
        The state object is updated to reflect the new state.

        As an example, a random trajectory of length horizon from the initial state could be implemented by the
        following (which also accumulates the reward along the trajectory).

        total_reward = 0
        for i in range(horizon):
            total_reward += sim_state.take_action(random_action())
        """
        raise NotImplementedError

    def update_current_player(self):
        self.current_player = (self.current_player + 1) % self.num_players

    @abc.abstractmethod
    def get_actions(self):
        """Returns the legal actions at the state."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_value_bounds(self):
        """Returns a dictionary specifying reward parameters.

        :return defeat: the reward incurred when an agent loses.
        :return victory: the reward incurred when an agent wins.
        :return min non-terminal: the lowest possible reward (excluding defeat).
        :return max non-terminal: the highest possible reward (excluding victory).
        :return pre-computed min: override value bound calculation with a pre-computed minimum value.
            None if not applicable.
        :return pre-computed max: override value bound calculation with a pre-computed maximum value.
            None if not applicable.
        :return evaluation function: a function for directly evaluating the bounds of particular states.
            Takes a game state object as input.
        """
        return {'defeat': None, 'victory': None,
                'min non-terminal': None, 'max non-terminal': None,
                'pre-computed min': float('-inf'), 'pre-computed max': float('inf'),
                'evaluation function': None}

    @abc.abstractmethod
    def is_terminal(self):
        """Returns true if in a terminal state."""
        raise NotImplementedError
