import abc
from abc import ABCMeta


class AbstractEvaluation:
    """The main class for implementing state evaluation functions."""
    __metaclass__ = ABCMeta
    name = ""

    @abc.abstractmethod
    def evaluate(self, state):
        """Using the predefined policy/policies, width, and depth, evaluates the state.

        :returns reward: a list [r1, ..., r_n] for an n-agent game.
        """
        raise NotImplementedError
