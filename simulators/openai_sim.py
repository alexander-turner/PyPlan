import time
import copy
import sys
import os
sys.path.append(os.path.abspath('simulators\\gym-master'))
from abstract import absstate
import gym
from gym import spaces
from gym import wrappers


class OpenAIStateClass(absstate.AbstractState):
    """An interface to run bandit algorithms on the OpenAI environment library.

    The simulator can be found at https://github.com/openai/gym.

    Note that unlike other interfaces, this is not compatible with the dealer simulator due its reliance on the openAI
    engine.
    """

    def __init__(self, sim_name, policy=None, wrapper_target='', api_key=''):
        """Initialize an interface with the specified OpenAI simulation task and policy.

        :param sim_name: a valid env key that corresponds to a particular game / task.
        :param policy: an instantiated policy (ex: uniform rollout agent).
        :param wrapper_target: optional target filename for algorithm performance.
        :param api_key: API key for uploading results to OpenAI Gym.
        """
        self.sim_name = sim_name

        self.env = gym.make(sim_name)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        if not isinstance(self.action_space, spaces.discrete.Discrete):
            raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(self.action_space, self))
        if not isinstance(self.observation_space, spaces.discrete.Discrete):
            raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(self.observation_space, self))

        self.original_observation = self.env.reset()  # initial observation
        self.current_observation = self.original_observation

        self.wrapper_target = wrapper_target
        self.api_key = api_key
        if self.wrapper_target != '':  # set where the results will be written to
            self.wrapper_target = 'simulators\\gym-master\\results\\' + self.wrapper_target + \
                                  time.strftime("%m-%d_%I-%M-%S", time.gmtime())
            self.env = wrappers.Monitor(self.env, self.wrapper_target, write_upon_reset=False)  # TODO: mute logger output when enbaled

        self.done = False  # indicates if the current observation is terminal

        self.policy = policy
        if self.policy is None:
            self.agent = RandomAgent(self.action_space)
        else:
            self.agent = Agent(self.policy, self)

    def initialize(self):
        """Reinitialize the environment."""
        self.current_observation = self.env.reset()
        self.done = False

    def run(self):
        self.initialize()
        while self.done is False:
            action = self.agent.act()
            self.current_observation, _, self.done, _ = self.env.step(action)
            self.env.render()
        self.env.close()
        if self.wrapper_target != '':
            print('Episode finished after {} timesteps.'.format(self.env.get_total_steps()))
            if self.api_key != '':
                gym.upload(self.wrapper_target, api_key=self.api_key)

    def clone(self):
        new_sim = copy.deepcopy(self)
        new_sim.env = new_sim.env.unwrapped
        new_sim.wrapper_target = ''
        new_sim.api_key = ''
        #new_sim = OpenAIStateClass(self.sim_name, self.agent)
        #new_sim.set(self)
        return new_sim

    def number_of_players(self):  # TODO: FIX
        return 1

    def set(self, sim):
        self.env = sim.env  # TODO: this doesn't appear to work - changes first-level object
        self.current_observation = sim.current_observation
        self.done = sim.done

    def is_terminal(self):
        return self.done

    def get_current_player(self):  # TODO: FIX
        """Returns one-indexed index of current player (for compatibility with existing bandit library)."""
        return 1

    def take_action(self, action):
        """Take the action and update the current state accordingly."""
        self.current_observation, reward, self.done, _ = self.env.step(action)
        rewards = [-1 * reward] * self.number_of_players()  # reward other agents get
        rewards[0] *= -1  # correct agent reward
        return rewards

    def get_actions(self):  # TODO: narrow scope to current space? Fix for continuous action spaces?
        return range(self.action_space.n)

    def __hash__(self):
        return self.current_observation.__hash__()


class Agent(object):
    """An agent that interfaces between a policy and the OpenAI environment."""
    def __init__(self, policy, parent):
        self.policy = policy
        self.parent = parent  # pointer to the agent's parent structure

    def act(self):
        return self.policy.select_action(self.parent)  # extra output coming from here


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()
