import time

import progressbar
import tabulate
import multiprocessing
import logging
import gym
from abstract import abstract_agent, abstract_dealer
from dealers.simulators import openai_sim


class Dealer(abstract_dealer.AbstractDealer):
    def __init__(self, env_name=None, force=True, api_key=None):
        """An object for running agents on environments.

        :param env_name: a valid env key that corresponds to a particular game / task.
        :param force: whether an existing directory at demos/OpenAI results/wrapper_target/ should be overwritten
        :param api_key: API key for uploading results to OpenAI Gym. Note that submissions are only scored if they are
            run for at least 100 trials.
        """
        self.change_wrapper(env_name)
        self.force = force
        self.api_key = api_key

        self.num_trials = 0
        self.multiprocess_mode = ''
        self.upload = False
        self.resume = False  # whether to add data to the output directory

        self.simulator = openai_sim.OpenAIState(dealer=self, env_name=env_name)

    def change_wrapper(self, env_name):
        self.env_name = env_name
        if self.env_name:
            # Output directory location for agent performance
            self.wrapper_target = 'OpenAI results\\' + self.env_name  # cut off version name

    def run_all(self, agents, num_trials, multiprocess_mode='trials', show_moves=False):
        """Runs the agents on all available simulators."""
        all_environments = self.available_configurations()
        for env_id in all_environments:
            if self.should_skip(env_id):  # duplicate games / games that hang
                print("Filtered game - skipping {}.".format(env_id))
                continue

            self.run(agents=agents, num_trials=num_trials, env_name=env_id,
                     multiprocess_mode=multiprocess_mode, show_moves=show_moves)

    def run(self, agents, num_trials, env_name=None, multiprocess_mode='trials', show_moves=True, upload=False):
        """Run the given number of trials on the specified agents, comparing their performance.

        Returns nothing if the environment's action space is continuous.

        :param agents: the agents with which to run trials. Should be instances of AbstractAgent.
        :param num_trials: how many trials to be run.
        :param env_name: the name of the environment to be run.
        :param multiprocess_mode: 'trials' for trial-wise multiprocessing, 'bandit' to multiprocess bandit arm pulls.
            other options will mean no multiprocessing is executed. Some games (such as Space Invaders) are not compatible
            with multiprocessing.
        :param show_moves: if the environment can render, then render each move.
        :param upload: whether to upload results to OpenAI.
        """
        self.num_trials = num_trials
        self.multiprocess_mode = multiprocess_mode
        self.show_moves = show_moves
        self.upload = upload

        if not env_name:
            env_name = self.env_name
        else:
            self.change_wrapper(env_name)

        try:
            self.simulator.load_env(env_name)
        except ValueError as e:  # If the action space is continuous
            logging.warning(e)
            return

        if self.multiprocess_mode == 'trials':  # doesn't make sense to show moves while multiprocessing
            self.show_moves = False

        for agent in agents:
            if not isinstance(agent, abstract_agent.AbstractAgent):  # if this isn't a planning agent
                self.multiprocess_mode = ''
                logging.warning('Multiprocessing is disabled for agents that learn over multiple episodes.')

        if not self.api_key:  # can't upload if we don't have an API key!
            self.upload = False

        table = []
        headers = ["Agent Name", "Average Episode Reward", "Success Rate", "Average Time / Move (s)"]
        if self.upload:
            headers.append("Link")

        if multiprocess_mode == 'trials':
            multiprocessing_str = "Trial-based"
        elif multiprocess_mode == 'bandit':
            multiprocessing_str = "Bandit-based"
        else:
            multiprocessing_str = "No"

        for agent in agents:
            print('\nNow simulating {}'.format(agent.agent_name))

            output = self.run_trials(agent)

            row = [agent.agent_name,  # agent name
                   output['average reward'],  # average final score
                   output['success rate'],  # win percentage
                   output['average move time']]  # average time taken per move
            if self.upload:
                row.append(output['url'])
            table.append(row)
        print("\n" + tabulate.tabulate(table, headers, tablefmt="grid", floatfmt=".4f"))
        print("Each agent ran {} game{} of {}. {} multiprocessing was used.".format(num_trials,
                                                                                    "s" if num_trials > 1 else "",
                                                                                    self.env_name[:-3],
                                                                                    multiprocessing_str))

    def run_trials(self, agent):
        """Run the given number of trials using the agent and the current configuration.

        :param agent: the agent to be used in the trials.
        """
        self.simulator.set_agent(agent)

        # We need to reinitialize the Monitor for the new trials we are about to run
        if not self.simulator.env.enabled:
            self.simulator.env = gym.make(self.env_name)
            self.simulator.env = gym.wrappers.Monitor(self.simulator.env, self.wrapper_target,
                                                      write_upon_reset=True, force=self.force, resume=self.resume)

        game_outputs = []
        if self.multiprocess_mode == 'trials':
            self.resume = True
            game_outputs = self.try_pool()
            self.resume = False  # done adding to the data
        else:
            if self.multiprocess_mode == 'bandit' and hasattr(agent, 'set_multiprocess'):
                old_config = agent.multiprocess
                agent.set_multiprocess(True)

            bar = progressbar.ProgressBar(max_value=self.num_trials)
            for i in bar(range(self.num_trials)):
                game_outputs.append(self.run_trial(i))
            time.sleep(0.1)  # so we don't print extra progress bars

            if self.multiprocess_mode == 'bandit' and hasattr(agent, 'set_multiprocess'):
                agent.set_multiprocess(old_config)

        total_reward, wins, total_time, total_steps = 0, 0, 0, 0
        stats_recorder = self.simulator.env.stats_recorder
        for output in game_outputs:
            total_reward += output['reward']
            wins += output['won']
            total_time += output['total time']
            total_steps += output['episode length']

            # Record episode information
            if self.multiprocess_mode == 'trials':
                stats_recorder.timestamps.append(output['timestamp'])
                stats_recorder.episode_lengths.append(output['episode length'])
                stats_recorder.episode_rewards.append(output['reward'])
                stats_recorder.episode_types.append(output['episode type'])

        stats_recorder.flush()

        self.simulator.env.close()
        url = None
        if self.upload:
            url = gym.upload(self.wrapper_target, api_key=self.api_key)

        return {'average reward': total_reward / self.num_trials,
                'success rate': wins / self.num_trials,
                'average move time': total_time / total_steps,
                'url': url}

    def try_pool(self):
        """Sometimes the pool takes a few tries to execute; keep trying until it works."""
        # Ensures the system runs smoothly
        pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1))
        while True:
            try:
                game_outputs = pool.map(self.run_trial, range(self.num_trials))
                pool.close()
                return game_outputs
            except WindowsError:
                pass
            except TypeError:  # encountered a thread.Lock in video recorder
                self.simulator.env.video_recorder = None  # we aren't displaying moves during multiprocessing anyways
            except ValueError:  # encountered a ctypes object with pointers
                if str.startswith(self.env_name, "CartPole"):
                    self.simulator.env.video_recorder = None
                    self.simulator.env.env.unwrapped.viewer = None

    def run_trial(self, trial_num):
        """Using the game parameters, run and return total time spent selecting moves.

        :param trial_num: a placeholder parameter for compatibility with multiprocessing.Pool.
        """
        self.simulator.reinitialize()

        total_time = 0
        while not self.simulator.is_terminal():
            begin = time.time()
            action = self.simulator.agent.act()
            total_time += time.time() - begin

            rewards = self.simulator.take_action(action)

            # Don't render if it's not supported
            if self.show_moves and 'human' in self.simulator.env.metadata['render.modes']:
                self.simulator.env.render()

        stats_recorder = self.simulator.env.stats_recorder
        return {'reward': stats_recorder.rewards,
                'won': rewards[0] > 0,  # the agent won if last reward observed > 0
                'total time': total_time,
                'episode length': stats_recorder.total_steps,
                'timestamp': stats_recorder.timestamps[0],
                'episode type': stats_recorder.type}

    @staticmethod
    def available_configurations():
        """Lists all available environments.

        To print nicely, use the pprint module.
        """
        configurations = []
        for e in gym.envs.registry.all():
            configurations.append(e.id)
        configurations.sort()
        return configurations

    @staticmethod
    def should_skip(game_id):
        """Returns True if the game should be skipped."""
        skip_games = ["Assault", "BankHeist", "BeamRider-v4", "CartPole", "CliffWalking"]  # problematic games
        skip_keywords = ["Deterministic", "Frameskip", "ram"]
        for key in skip_games + skip_keywords:
            if str.find(game_id, key) != -1:
                return True
