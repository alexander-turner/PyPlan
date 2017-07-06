import logging
import copy
import numpy
import itertools
import tabulate
import multiprocessing
import gym
import time
from gym import spaces
from gym import wrappers
from abstract import absstate, absagent


class Dealer:
    def __init__(self, sim_name=None, force=True, api_key=None):
        """Initialize a Dealer object for running trials of agents on environments.

        :param sim_name: a valid env key that corresponds to a particular game / task.
        :param force: whether an existing directory at demos/OpenAI results/wrapper_target/ should be overwritten
        :param api_key: API key for uploading results to OpenAI Gym. Note that submissions are only scored if they are
            run for at least 100 trials.
        """
        self.change_sim_path(sim_name)
        self.force = force
        self.api_key = api_key
        self.resume = False  # whether to add data to the output directory
        self.simulator = OpenAIStateClass(dealer=self, sim_name=sim_name)

    def change_sim_path(self, sim_name):
        self.sim_name = sim_name
        if self.sim_name:
            self.wrapper_target = 'OpenAI results\\' + self.sim_name[:-3]  # cut off version name

    def run_all(self, agents, num_trials, multiprocess=False, show_moves=False):
        """Runs the agents on all available simulators."""
        all_environments = gym.envs.registry.all()
        for env_idx, env in enumerate(all_environments):
            if self.should_skip(env.id):  # duplicate games / games that hang
                print("Filtered game - skipping {}.".format(env.id))
                continue

            self.run(sim_name=env.id, agents=agents, num_trials=num_trials, multiprocess=multiprocess,
                     show_moves=show_moves)

    def run(self, sim_name, agents, num_trials, multiprocess=True, show_moves=True, upload=False):
        """Run the given number of trials on the specified agents, comparing their performance.

        :param sim_name: the simulation to run.
        :param agents: the agents with which to run trials. Should be instances of AbstractAgent.
        :param num_trials: how many trials to be run.
        :param multiprocess: whether to use multiprocessing. Some games (such as Space Invaders) are not compatible
            with multiprocessing.
        :param show_moves: if the environment can render, then render each move.
        :param upload: whether to upload results to OpenAI.
        """
        self.change_sim_path(sim_name)
        try:
            self.simulator.load_env(sim_name)
        except ValueError:  # If the action space is continuous
            print("Continuous action space - skipping {}.".format(sim_name))
            return

        for agent in agents:
            agent = Agent(agent, self.simulator)
            if not isinstance(agent.policy, absagent.AbstractAgent):  # if this isn't a planning agent
                multiprocess = False
                logging.warning('Multiprocessing is disabled for agents which learn over multiple episodes.')

        if multiprocess:  # doesn't make sense to show moves while multiprocessing
            show_moves = False  # TODO: investigate why it doesn't work with Space Invaders

        if not self.api_key:  # can't upload if we don't have an API key!
            upload = False

        self.show_moves = show_moves

        table = []
        headers = ["Agent Name", "Average Episode Reward", "Success Rate", "Average Time / Move (s)"]
        if upload:
            headers.append("Link")

        for agent in agents:
            print('\nNow simulating: {}'.format(agent.agent_name))
            output = self.run_trials(agent, num_trials, multiprocess, upload)
            row = [output['name'],  # agent name
                   output['average reward'],  # average final score
                   output['success rate'],  # win percentage
                   output['average move time']]  # average time taken per move
            if upload:
                row.append(output['url'])
            table.append(row)
        print("\n" + tabulate.tabulate(table, headers, tablefmt="grid", floatfmt=".4f"))
        print("Each agent ran {} game{} of {}.".format(num_trials, "s" if num_trials > 1 else "",
                                                       self.simulator.my_name))

    def run_trials(self, agent, num_trials=1, multiprocess=True, upload=False):
        """Run the given number of trials using the current configuration."""
        self.simulator.set_agent(agent)

        # We need to reinitialize the Monitor for the new trials we are about to run
        if not self.simulator.env.enabled:
            self.simulator.env = gym.make(self.simulator.sim_name)
            self.simulator.env = wrappers.Monitor(self.simulator.env, self.wrapper_target,
                                                  write_upon_reset=True, force=self.force, resume=self.resume)

        game_outputs = []
        if multiprocess:
            self.resume = True
            game_outputs = self.try_pool(num_trials)
            self.resume = False  # done adding to the data
        else:
            for i in range(num_trials):
                game_outputs.append(self.run_trial(i))

        total_reward, wins, total_time, total_steps = 0, 0, 0, 0
        stats_recorder = self.simulator.env.stats_recorder
        for output in game_outputs:
            total_reward += output['reward']
            wins += output['won']
            total_time += output['total time']
            total_steps += output['episode length']

            # Record episode information
            if multiprocess:
                stats_recorder.timestamps.append(output['timestamp'])
                stats_recorder.episode_lengths.append(output['episode length'])
                stats_recorder.episode_rewards.append(output['reward'])
                stats_recorder.episode_types.append(output['episode type'])

        stats_recorder.flush()

        self.simulator.env.close()
        url = None
        if self.api_key and upload:
            url = gym.upload(self.wrapper_target, api_key=self.api_key)

        return {'name': agent.agent_name, 'average reward': total_reward / num_trials,
                'success rate': wins / num_trials,
                'average move time': total_time / total_steps, 'url': url}

    def try_pool(self, num_trials):
        """Sometimes the pool takes a few tries to execute; keep trying until it works."""
        # Ensures the system runs smoothly
        pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1))
        while True:
            try:
                return pool.map(self.run_trial, range(num_trials))
            except WindowsError:
                pass

    def run_trial(self, trial_num):
        """Using the game parameters, run and return total time spent selecting moves.

        :param trial_num: a placeholder parameter for compatibility with multiprocessing.Pool.
        """
        self.simulator.reinitialize()  # #TODO: Initialize simulator for each thread? https://stackoverflow.com/questions/41071563/python-multiprocessing-cant-pickle-thread-lock-pymongo
        #self.simulator.env.video_recorder.enabled = self.show_moves
        total_time = 0
        while not self.simulator.is_terminal():
            begin = time.time()
            action = self.simulator.agent.act()
            total_time += time.time() - begin

            rewards = self.simulator.take_action(action)

            # Don't render if not supported
            if self.show_moves and 'human' in self.simulator.env.metadata['render.modes']:
                self.simulator.env.render()  # TODO: investigate CartPole-v0

        stats_recorder = self.simulator.env.stats_recorder
        return {'reward': stats_recorder.rewards,
                'won': rewards[0] > 0,  # won if reward after game ends is positive
                'total time': total_time,
                'episode length': stats_recorder.total_steps,
                'timestamp': stats_recorder.timestamps[0],
                'episode type': stats_recorder.type}

    @staticmethod
    def should_skip(game_id):
        """Returns True if the game should be skipped."""
        skip_games = ["Assault", "BankHeist", "BeamRider-v4", "CliffWalking"]  # games take a very long time
        skip_keywords = ["Deterministic", "Frameskip", "ram"]
        for key in skip_games + skip_keywords:
            if str.find(game_id, key) != -1:
                return True


class OpenAIStateClass(absstate.AbstractState):
    """An interface to run bandit algorithms on the OpenAI environment library.

    The simulator can be found at https://github.com/openai/gym.

    Note that unlike other interfaces, this is not compatible with the dealer simulator due its reliance on the openAI
    engine.
    """

    def __init__(self, dealer, sim_name=None):
        """Initialize an interface with the specified OpenAI simulation task and policy.

        :param dealer: a Dealer object that can be accessed runtime parameter information.
        :param sim_name: a valid env key that corresponds to a particular game / task.
        """
        self.agent = None
        self.dealer = dealer
        if sim_name:
            self.load_env(sim_name)

    def reinitialize(self):
        """Reinitialize the environment."""
        self.current_observation = self.env.reset()
        self.done = False

    def clone(self):
        new_sim = copy.copy(self)
        new_sim.env = new_sim.env.unwrapped
        return new_sim

    def set(self, sim):
        self.env = copy.copy(sim.env.unwrapped)
        self.current_observation = sim.current_observation
        self.done = sim.done

    def set_agent(self, agent):
        self.agent = Agent(agent, self)

    def load_env(self, sim_name):
        self.sim_name = sim_name
        self.env = gym.make(sim_name)
        self.my_name = self.env.spec._env_name

        # output directory location for agent performance
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
