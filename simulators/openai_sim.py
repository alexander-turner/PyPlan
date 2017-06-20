import sys
import os
sys.path.append(os.path.abspath('simulators\\gym-master'))
from abstract import absstate
import gym


class OpenAIStateClass(absstate.AbstractState):
    """An interface to run bandit algorithms on the OpenAI environment library.

    The simulator can be found at https://github.com/openai/gym.

    Note that unlike other interfaces, this is not compatible with the dealer simulator due its reliance on the openAI
    engine.
    """

    def __init__(self, sim_name, policy=None):
        """

        :param sim_name: a valid env key that corresponds to a particular game / task.
        """
        self.sim_name = sim_name
        self.env = gym.make(sim_name)
        self.current_observation = self.env.reset()  # initial observation

        self.done = False  # indicates if the current observation is terminal
        self.reward = 0

        self.steps_elapsed = 0  # how many steps the agent has used

        self.policy = policy
        if self.policy is None:
            self.agent = RandomAgent(self.env.action_space)
        else:
            self.agent = Agent(self.env.action_space, self.policy)

    def initialize(self):
        """Reinitialize using the defined layout, Pacman agent, ghost agents, and display."""
        self.current_observation = self.env.reset()
        self.done = False
        self.reward = 0
        self.steps_elapsed = 0

    def run(self):
        self.done = False
        self.steps_elapsed = 0
        while self.done is False:
            self.env.render()
            action = self.agent.act(self.current_observation, self.reward, self.done)
            self.current_observation, self.reward, self.done, _ = self.env.step(action)
            self.steps_elapsed += 1

    def clone(self):  # TODO: FIX
        new_sim_obj = OpenAIStateClass(self.sim_name, self.agent)
        new_sim_obj.done = self.done
        return new_sim_obj

    def number_of_players(self):  # TODO: FIX
        return 1

    def set(self, sim):  # TODO: FIX
        self.current_observation = sim.current_observation

    def is_terminal(self):  # TODO: FIX
        return self.done

    def get_current_player(self):  # TODO: FIX
        """Returns one-indexed index of current player (for compatibility with existing bandit library)."""
        return 1

    def take_action(self, action):  # TODO: FIX
        """Take the action and update the current state accordingly."""
        return [0]

    def get_actions(self):  # TODO: FIX
        return self.env.action_space()

    def __hash__(self):
        return self.current_observation.__hash__()


class Agent(object):
    """An agent that interfaces between a policy and the OpenAI environment."""
    def __init__(self, action_space, policy, parent):
        self.action_space = action_space
        self.policy = policy
        self.parent = parent  # pointer to the agent's parent structure

    def act(self, observation, reward, done):
        #self.parent.current_observation = observation  # update state?
        return self.policy.select_action(self.parent)


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
