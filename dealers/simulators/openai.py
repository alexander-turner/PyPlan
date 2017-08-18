import copy
import numpy
import itertools
import gym
from gym import spaces
from gym import wrappers
from abstract import abstract_state


class OpenAIState(abstract_state.AbstractState):
    """An interface to run bandit algorithms on the OpenAI Gym environment library."""
    num_players = 1
    current_player = 0

    def __init__(self, dealer, env_name=None):
        """Initialize an interface with the specified OpenAI simulation task and policy.

        :param dealer: a Dealer object which provides runtime parameter information.
        :param env_name: a valid env key that corresponds to a particular game / task.
        """
        self.dealer = dealer
        if env_name:
            self.load_env(env_name)

    def reinitialize(self):
        """Reinitialize the environment."""
        self.current_observation = self.env.reset()
        self.done = False

    def clone(self):
        new_sim = copy.copy(self)
        new_sim.env = copy.deepcopy(self.env.unwrapped)
        return new_sim

    def set(self, sim):
        self.env = copy.copy(sim.env.unwrapped)
        self.current_observation = sim.current_observation
        self.done = sim.done

    def set_agent(self, agent):
        self.agent = agent

    def load_env(self, env_name):
        self.sim_name = env_name
        self.env = gym.make(env_name)
        self.env_name = self.env.spec._env_name

        self.env = wrappers.Monitor(self.env, self.dealer.wrapper_target, write_upon_reset=True,
                                    force=self.dealer.force,
                                    resume=self.dealer.resume)

        self.action_space = self.env.action_space
        # Check that we can handle the action space
        if (not isinstance(self.action_space, spaces.Discrete) and not isinstance(self.action_space, spaces.Tuple)) or \
                (hasattr(self.action_space, 'spaces') and self.contains_type(self.action_space.spaces, spaces.Box)):
            raise ValueError('Action space {} incompatible with {} (only supports Discrete and Tuple action spaces).'
                             .format(self.action_space, self))
        self.observation_space = self.env.observation_space

        self.current_observation = self.env.reset()  # initial observation
        self.done = False  # indicates if the current observation is terminal

    @staticmethod
    def contains_type(lst, t):
        """Returns true if at least one element in lst is of type t."""
        for l in lst:
            if isinstance(l, t):
                return True

    def take_action(self, action):
        """Take the action and update the current state accordingly."""
        self.current_observation, reward, self.done, _ = self.env.step(action)
        return [reward]

    def get_actions(self):
        if isinstance(self.action_space, spaces.Discrete):
            return range(self.action_space.n)
        elif isinstance(self.action_space, spaces.Tuple):
            action_spaces = self.action_space.spaces
            ranges = tuple(tuple(range(s.n)) for s in action_spaces)
            product = tuple(itertools.product(*ranges))  # return all combinations of action dimensions
            return product
        else:
            raise NotImplementedError

    def get_value_bounds(self):
        """Environments only expose the absolute reward range, so we don't differentiate if it's terminal or not."""
        return {'defeat': self.env.reward_range[0], 'victory': self.env.reward_range[1],
                'min non-terminal': self.env.reward_range[0], 'max non-terminal': self.env.reward_range[1],
                'pre-computed min': None, 'pre-computed max': None,
                'evaluation function': None}

    def is_terminal(self):
        return self.done

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(self.current_observation.data.tobytes()) if isinstance(self.current_observation, numpy.ndarray) \
            else hash(self.current_observation)
