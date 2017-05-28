import abc
from abc import ABCMeta

# This is the main class for implementing simulators

"""

In addition to the below methods it is important to implement _eq_ and _hash_ which is used by some planning
methods to judge equivalence of states. If these are not implemented then the code will work, but will perhaps
be sub-optimal due to the fact that states that are fundamentally the same, but correspond to distinct objects will
be treated as non-equivalent.

"""

class AbstractState:
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def initialize(self):
        """

        This method sets the state to the/an initial state of the domain

        """
        raise NotImplementedError

    @abc.abstractmethod
    def number_of_players(self):
        """

        This method returns the number of players

        """

        raise NotImplementedError

    @abc.abstractmethod
    def take_action(self, action):
        """

        This method simulates the result of taking action in the state. It returns the resulting reward vector,
        where the reward is a list [r1,...,r_n] for an n agent game. The state object is updated to reflect the new
        state.

        As an example, a random trajectory of length horizion from the initial state could be implemented by the
        following which also accumulates the reward along the trajectory.

        total_reward = 0
        s.initialize()
        for i in range(horizon):
            total_reward += s.take_action(random_action())

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_actions(self):
        """

        Returns the legal actions at the state.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def clone(self):
        """

        Creates a deep copy of the state

        """
        raise NotImplementedError

    @abc.abstractmethod
    def set(self,state):
        """

        Makes the object equivalent to state by copying the critical information from state

        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_terminal(self):
        """

        Returns true if the object is a terminal state

        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_current_player(self):
        """

        Returns the index of the current player.
        Index values are in the range {1,...,num_players}.

        It is important to note that the first player is 1 rather than 0. It may be advised to change this,
        but the current planners are using the 1,... range.

        """
        raise NotImplementedError
