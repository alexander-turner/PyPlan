import abc
from abc import ABCMeta


class AbstractHeuristic:
    """The main class for implementing heuristics."""
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def get_heuristic_name(self):
        """Returns the name of the heuristic."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, state):
        """Using the predefined policy/policies, width, and depth, evaluates the state.

        :returns reward:, a list [r1,...,r_n] for an n agent game.
        """
        raise NotImplementedError
