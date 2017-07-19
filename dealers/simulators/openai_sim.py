import copy
import numpy
import itertools
import gym
from gym import spaces
from gym import wrappers
from abstract import abstract_state


class OpenAIState(abstract_state.AbstractState):
    """An interface to run bandit algorithms on the OpenAI environment library.

    The simulator can be found at https://github.com/openai/gym.

    Note that unlike other interfaces, this is not compatible with the dealer simulator due its reliance on the openAI
    engine.
    """

    def __init__(self, dealer, env_name=None):
        """Initialize an interface with the specified OpenAI simulation task and policy.

        :param dealer: a Dealer object which provides runtime parameter information.
        :param env_name: a valid env key that corresponds to a particular game / task.
        """
        self.dealer = dealer
        if env_name:
            self.load_env(env_name)
        self.agent = None

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
        self.agent = Agent(agent, self)

    def load_env(self, env_name):
        self.sim_name = env_name
        self.env = gym.make(env_name)
        self.my_name = self.env.spec._env_name

        self.env = wrappers.Monitor(self.env, self.dealer.wrapper_target, write_upon_reset=True, force=self.dealer.force,
                                    resume=self.dealer.resume)

        self.action_space = self.env.action_space
        if not isinstance(self.action_space, spaces.Discrete) and not isinstance(self.action_space, spaces.Tuple):
            raise ValueError('Action space {} incompatible with {} (only supports Discrete and Tuple action spaces).'
                             .format(self.action_space, self))
        self.observation_space = self.env.observation_space

        self.current_observation = self.env.reset()  # initial observation
        self.done = False  # indicates if the current observation is terminal

    def number_of_players(self):
        return 1

    def is_terminal(self):
        return self.done

    def get_current_player(self):
        """Returns index of current player."""
        return 0

    def take_action(self, action):
        """Take the action and update the current state accordingly."""
        self.current_observation, reward, self.done, _ = self.env.step(action)
        rewards = [-1 * reward] * self.number_of_players()  # reward other agents get
        rewards[0] *= -1  # correct agent reward
        return rewards

    def get_actions(self):
        if isinstance(self.action_space, spaces.Discrete):
            return range(self.action_space.n)
        elif isinstance(self.action_space, spaces.Tuple):
            action_spaces = self.action_space.spaces
            ranges = tuple(tuple(range(s.n)) for s in action_spaces)  # TODO: take advantage of tuple aspect
            product = tuple(itertools.product(*ranges))  # return all combinations of action dimensions
            return product
        else:
            raise NotImplementedError

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        if isinstance(self.current_observation, numpy.ndarray):
            return hash(self.current_observation.data.tobytes())
        else:
            return hash(self.current_observation)


class Agent(object):
    """An agent that interfaces between a policy and the OpenAI environment."""
    def __init__(self, policy, parent):
        self.policy = policy
        self.parent = parent  # pointer to the agent's parent structure

    def act(self):
        """Despite what its name may suggest, act only determines what action to take."""
        return self.policy.select_action(self.parent)