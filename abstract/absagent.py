import abc
from abc import ABCMeta


class AbstractAgent:
    """The main class for implementing agents.

    Agents act as the interface between the simulator object and the bandit algorithm.

    For help making your own agent, reference the included examples.
    """
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def select_action(self, state):
        """Return the highest-valued action for the given state."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_agent_name(self):
        """Return the agent name."""
        raise NotImplementedError
