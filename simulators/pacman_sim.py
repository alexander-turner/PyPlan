import sys
import os

sys.path.append(os.path.abspath('..\\simulators\\pacmancode'))  # assume access from demos\
from abstract import absstate
from simulators.pacmancode import *
from functools import partial
import time
import tabulate
import numpy
import multiprocessing


class PacmanStateClass(absstate.AbstractState):
    """An interface to run bandit algorithms on the Pacman simulator provided by Berkeley.

    Give multiple agents to compare results.

    The simulator can be found at http://ai.berkeley.edu/project_overview.html.

    Note that unlike other interfaces, this is not compatible with the dealer simulator due its reliance on the provided
    Pacman engine.
    """

    def __init__(self, layout_repr, use_random_ghost=False, use_graphics=True):
        """Initialize an interface to the Pacman game simulator.

        :param layout_repr: the filename of the layout (located in layouts/), or an actual layout object.
        :param use_random_ghost: whether to use the random or the directional ghost agent.
        :param use_graphics: whether to use the graphics or the text display.
        """
        if isinstance(layout_repr, str):
            self.layout = layout.getLayout(layout_repr)
        else:  # we've been directly given a layout
            self.layout = layout_repr
        self.my_name = "Pacman"

        if use_graphics:
            self.display = graphicsDisplay.PacmanGraphics()
        else:
            self.display = textDisplay.PacmanGraphics()
        self.show_moves = True

        if use_random_ghost:
            self.ghost_agents = [ghostAgents.RandomGhost(i) for i in range(1, self.layout.getNumGhosts() + 1)]
        else:
            self.ghost_agents = [ghostAgents.DirectionalGhost(i) for i in range(1, self.layout.getNumGhosts() + 1)]

        self.pacman_agent = None

        self.game = pacman.ClassicGameRules.newGame(self, layout=self.layout, pacmanAgent=self.pacman_agent,
                                                    ghostAgents=self.ghost_agents, display=self.display)
        self.current_state = self.game.state
        self.num_players = self.game.state.getNumAgents()

        self.final_score = float("-inf")  # demarcates we have yet to obtain a final score
        self.won = False
        self.time_step_count = 0

    def reinitialize(self):
        """Reinitialize using the defined layout, Pacman agent, ghost agents, and display."""
        self.game = pacman.ClassicGameRules.newGame(self, layout=self.layout, pacmanAgent=self.pacman_agent,
                                                    ghostAgents=self.ghost_agents, display=self.display)
        self.current_state = self.game.state
        self.final_score = float("-inf")
        self.won = False
        self.time_step_count = 0  # how many total turns have elapsed

    def run(self, agents, num_trials=1, multiprocess=True, show_moves=False):
        """Runs num_trials trials for each of the provided agents, neatly displaying results (if requested)."""
        self.show_moves = show_moves  # whether game moves should be shown

        table = []
        headers = ["Agent Name", "Average Final Score", "Winrate", "Average Time / Move (s)"]

        for agent in agents:
            print('\nNow simulating: {}'.format(agent.agent_name))
            output = self.run_trials(agent, num_trials, multiprocess)
            table.append([output['name'],  # agent name
                          numpy.mean(output['rewards']),  # average final score
                          output['wins'] / num_trials,  # win percentage
                          output['average move time']])  # average time taken per move
        print("\n" + tabulate.tabulate(table, headers, tablefmt="grid", floatfmt=".4f"))
        print("Each agent ran {} game{} of {}.".format(num_trials, "s" if num_trials > 1 else "", self.my_name))

    def run_trials(self, agent, num_trials, multiprocess=True):
        """Run a given number of games using the current configuration, recording and returning performance statistics.

        :param agent: an agent to use to run the trials.
        :param num_trials: how many times the game will be run.
        :param multiprocess: whether to speed the computation with parallel processing.
        """
        self.set_agent(agent)

        game_outputs = []
        if multiprocess:
            # ensures the system runs smoothly
            pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1))
            game_outputs = pool.map(self.run_trial, range(num_trials))
        else:
            for i in range(num_trials):
                game_outputs.append(self.run_trial(i))

        rewards = []
        wins, average_move_time = 0, 0
        for output in game_outputs:
            rewards.append(output['reward'])
            wins += output['won']
            average_move_time += output['average move time']

        average_move_time /= num_trials

        return {'name': self.pacman_agent.policy.agent_name, 'rewards': rewards, 'wins': wins,
                'average move time': average_move_time}

    def run_trial(self, trial_num):
        """Using the game parameters, run and return information about the trial.

        :param trial_num: a placeholder parameter for compatibility with multiprocessing.Pool.
        """
        self.reinitialize()  # reset the game

        start_time = time.time()
        self.game.run(self.show_moves)
        time_taken = time.time() - start_time

        return {'reward': self.final_score, 'won': self.won, 'average move time': time_taken / self.time_step_count}

    def set_agent(self, agent):
        """Sets Pacman's agent."""
        self.pacman_agent = Agent(agent, self)
        self.reinitialize()

    def clone(self):
        new_sim = PacmanStateClass(self.layout)
        new_sim.current_state = self.current_state
        return new_sim

    def number_of_players(self):
        return self.num_players

    def set(self, sim):
        self.current_state = sim.current_state

    def is_terminal(self):
        return self.current_state.isWin() or self.current_state.isLose()

    def process(self, state, game):
        """Wrapper to help with ending the game."""
        if state.isWin():
            self.final_score = state.data.score
            self.won = True
            pacman.ClassicGameRules.win(self, state=state, game=game)
        if state.isLose():
            self.final_score = state.data.score
            pacman.ClassicGameRules.lose(self, state=state, game=game)

    def get_current_player(self):
        """Pacman is the only player."""
        return 0

    def take_action(self, action):
        """Take the action and update the current state accordingly."""
        new_state = self.current_state.generateSuccessor(self.get_current_player(), action)  # simulate Pacman movement

        for ghostInd, ghost in enumerate(self.ghost_agents):  # simulate ghost movements
            if new_state.isWin() or new_state.isLose():
                break
            ghost_action = ghost.getAction(new_state)
            new_state = new_state.generateSuccessor(ghostInd + 1, ghost_action)

        reward = new_state.getScore() - self.current_state.getScore()  # reward Pacman gets
        rewards = [-1 * reward] * self.number_of_players()  # reward ghosts get
        rewards[0] *= -1  # correct Pacman reward

        self.current_state = new_state

        self.time_step_count += 1  # mark that another time step has elapsed

        return rewards  # how much our score increased because of this action

    def get_actions(self):
        return self.current_state.getLegalActions()

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(self.current_state)


class Agent(game.Agent):
    """A wrapper to let the policy interface with the Pacman engine."""

    def __init__(self, policy, pac_state):
        self.policy = policy
        self.pac_state = pac_state  # pointer to parent PacmanStateClass structure

    def getAction(self, state):
        self.pac_state.current_state = state
        self.pac_state.time_step_count += 1
        return self.policy.select_action(self.pac_state)
