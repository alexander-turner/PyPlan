import abc
from abc import ABCMeta


class AbstractState:
    """The main class for implementing simulators.

    In addition to the methods below, it is important to implement __eq__ and __hash__, which are used by some planning
    methods to judge equivalence of states. If these are not implemented, then the code will work, but will perhaps
    be sub-optimal. This is because states that are fundamentally the same but correspond to distinct
    objects will be treated as non-equivalent.

    Simulators compatible with native_dealer.py must implement set_current_player (if not single-player) and __str__.

    Simulators which are incompatible with native_dealer.py should have their own dealer, which must implement three
    additional methods: run, run_trials, and run_trial. In short, run handles data processing, initiates run_trials
    for each of the provided agents, and displays results. run_trials runs the specified number of trials on the
    given agent, recording pertinent statistics. run_trial runs a single iteration of the simulator with the given
    agent. multiprocessing compatibility is encouraged. openai_sim.py and pacman_sim.py contain implementation
    examples.
    """
    __metaclass__ = ABCMeta

    my_name = ""  # the name of the environment

    @abc.abstractmethod
    def reinitialize(self):
        """This method sets the state to the/an initial state of the domain."""
        raise NotImplementedError

    @abc.abstractmethod
    def number_of_players(self):
        """This method returns the number of players."""

        raise NotImplementedError

    @abc.abstractmethod
    def take_action(self, action):
        """This method simulates the result of taking action in the state.

        It returns the resulting reward vector,
        where the reward is a list [r1,...,r_n] for an n agent game. The state object is updated to reflect the new
        state.

        As an example, a random trajectory of length horizon from the initial state could be implemented by the
        following (which also accumulates the reward along the trajectory).

        total_reward = 0
        s.initialize()
        for i in range(horizon):
            total_reward += s.take_action(random_action())
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_actions(self):
        """Returns the legal actions at the state."""
        raise NotImplementedError

    @abc.abstractmethod
    def clone(self):
        """Creates a deep copy of the state."""
        raise NotImplementedError

    @abc.abstractmethod
    def set(self, state):
        """Makes the object equivalent to state by copying the critical information from state."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_terminal(self):
        """Returns true if the object is in a terminal state."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_current_player(self):
        """Returns the index of the current player.

        Index values are in the range {0,...,num_players-1}.
        """
        raise NotImplementedError
