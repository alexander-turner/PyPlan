import logging
import copy
import multiprocessing
import gym
from gym import spaces
from gym import wrappers
from abstract import absstate, absagent


class OpenAIStateClass(absstate.AbstractState):
    """An interface to run bandit algorithms on the OpenAI environment library.

    The simulator can be found at https://github.com/openai/gym.

    Note that unlike other interfaces, this is not compatible with the dealer simulator due its reliance on the openAI
    engine.
    """

    def __init__(self, sim_name, policy, wrapper_target='', api_key='', force=True):
        """Initialize an interface with the specified OpenAI simulation task and policy.

        :param sim_name: a valid env key that corresponds to a particular game / task.
        :param policy: an instantiated policy (ex: uniform rollout agent).
        :param wrapper_target: optional target filename for algorithm performance.
        :param api_key: API key for uploading results to OpenAI Gym. Note that submissions are only scored if they are
            run for at least 100 trials.
        :param force: whether an existing directory at demos/OpenAI results/wrapper_target/ should be overwritten
        """
        self.sim_name = sim_name
        self.env = gym.make(sim_name)  # the monitored environment - modified in actual run()
        self.myname = self.env.spec._env_name

        self.wrapper_target = wrapper_target
        self.api_key = api_key
        self.resume = False  # whether to add data to the wrapper target directory
        if self.wrapper_target != '':  # set where the results will be written to
            if not self.wrapper_target.startswith('OpenAI results\\'):
                self.wrapper_target = 'OpenAI results\\' + self.wrapper_target
            # Initialized from code at C:\Python36\Lib\site-packages\gym\wrappers\monitoring.py
            self.env = wrappers.Monitor(self.env, self.wrapper_target, write_upon_reset=True, force=force,
                                        resume=self.resume)

        self.action_space = self.env.action_space
        if not isinstance(self.action_space, spaces.discrete.Discrete):
            raise Exception('Action space {} incompatible with {} (only supports Discrete action spaces).'.format(self.action_space, self))
        self.observation_space = self.env.observation_space

        self.original_observation = self.env.reset()  # initial observation
        self.current_observation = self.original_observation

        self.done = False  # indicates if the current observation is terminal

        self.agent = Agent(policy, self)

    def initialize(self):
        """Reinitialize the environment."""
        self.current_observation = self.env.reset()
        self.done = False

    def run(self, num_trials=1, multiprocess=True, do_render=True):  # TODO: run multiple agents and compare
        """Run the given number of trials using the current configuration."""
        if not isinstance(self.agent.policy, absagent.AbstractAgent):  # if this is an actual learning agent
            multiprocess = False
            logging.warning('Multiprocessing is disabled for agents which learn over multiple episodes.')

        if multiprocess:  # TODO: debug multiprocessing for wrappers
            self.resume = True
            if multiprocess:
                # ensures the system runs smoothly
                pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1))
                pool.map(self.run_trial, [do_render] * num_trials)
            self.resume = False  # done adding to the data
        else:
            for i in range(num_trials):
                self.run_trial(do_render)

        self.env.close()

        if self.wrapper_target != '':
            print('Episode finished after {} timesteps.'.format(self.env.get_total_steps()))
            if self.api_key != '':
                gym.upload(self.wrapper_target, api_key=self.api_key)

    def run_trial(self, do_render):
        self.initialize()
        while self.done is False:
            action = self.agent.act()
            self.current_observation, _, self.done, _ = self.env.step(action)
            if do_render:
                self.env.render()

    def clone(self):
        new_sim = copy.deepcopy(self)
        return new_sim

    def number_of_players(self):
        return 1

    def set(self, sim):
        self.env = copy.deepcopy(sim.env.unwrapped)
        self.current_observation = sim.current_observation
        self.done = sim.done

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

    def get_actions(self):  # TODO: narrow scope to current observation?
        return range(self.action_space.n)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(self.current_observation)


class Agent(object):
    """An agent that interfaces between a policy and the OpenAI environment."""
    def __init__(self, policy, parent):
        self.policy = policy
        self.parent = parent  # pointer to the agent's parent structure

    def act(self):
        """Despite what its name may suggest, act only determines what action to take."""
        return self.policy.select_action(self.parent)  # extra output coming from here

