import sys
import os
sys.path.append(os.path.abspath('..\\simulators\\pacmancode'))  # assume we're running from demos/
from abstract import absstate
from simulators.pacmancode import *
import time
import tabulate
import numpy
import multiprocessing


class Dealer:
    def __init__(self, layout_representation, multiprocess=True, show_moves=False, use_graphics=True):
        """

        :param layout_representation: either the layout filename (located in layouts/) or an actual layout object.
        :param multiprocess: whether to speed the computation with parallel processing.
        :param show_moves: whether moves should be rendered. Disabled if multiprocess is True.
        :param use_graphics: whether to use the graphics or the text display.
        """
        self.multiprocess = multiprocess
        self.show_moves = show_moves
        self.use_graphics = use_graphics

        self.num_trials = 0

        self.simulator = PacmanState(dealer=self, layout_representation=layout_representation)

    def run(self, agents, num_trials=1, multiprocess=True, show_moves=False):
        """Runs num_trials trials for each of the provided agents, neatly displaying results (if requested).

        :param agents: the agents whose Pacman performance will be compared.
        :param num_trials: how many times the game will be run.
        :param multiprocess: whether to speed the computation with parallel processing.
        :param show_moves: whether moves should be rendered. Disabled if multiprocess is True.
        """
        self.show_moves = show_moves  # whether game moves should be shown
        self.num_trials = num_trials
        self.multiprocess = multiprocess

        table = []
        headers = ["Agent Name", "Average Final Score", "Winrate", "Average Time / Move (s)"]

        for agent in agents:
            print('\nNow simulating {}'.format(agent.agent_name))
            output = self.run_trials(agent)
            table.append([agent.agent_name,
                          numpy.mean(output['rewards']),  # average final score
                          output['wins'] / num_trials,  # win percentage
                          output['average move time']])
        print("\n" + tabulate.tabulate(table, headers, tablefmt="grid", floatfmt=".4f"))
        print("Each agent ran {} game{} of {}.".format(num_trials, "s" if num_trials > 1 else "",
                                                       self.simulator.my_name))

    def run_trials(self, agent):
        """Run a given number of games using the current configuration, recording and returning performance statistics.

        :param agent: the agent to be used in the trials.
        """
        self.simulator.set_agent(agent)

        game_outputs = []
        if self.multiprocess:
            # ensures the system runs smoothly
            pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1))
            game_outputs = pool.map(self.run_trial, range(self.num_trials))
        else:
            for i in range(self.num_trials):
                game_outputs.append(self.run_trial(i))

        rewards = []
        wins, average_move_time = 0, 0
        for output in game_outputs:
            rewards.append(output['reward'])
            wins += output['won']
            average_move_time += output['average move time']

        average_move_time /= self.num_trials

        return {'rewards': rewards, 'wins': wins, 'average move time': average_move_time}

    def run_trial(self, trial_num):
        """Using the game parameters, run and return information about the trial.

        :param trial_num: a placeholder parameter for compatibility with multiprocessing.Pool.
        """
        self.simulator.reinitialize()  # reset the game

        start_time = time.time()
        self.simulator.game.run(self.show_moves)
        time_taken = time.time() - start_time

        return {'reward': self.simulator.final_score, 'won': self.simulator.won,
                'average move time': time_taken / self.simulator.time_step_count}


class PacmanState(absstate.AbstractState):
    """An interface to run bandit algorithms on the Pacman simulator provided by Berkeley.

    Give multiple agents to compare results.

    The simulator can be found at http://ai.berkeley.edu/project_overview.html.

    Note that unlike other interfaces, this is not compatible with the dealer simulator due its reliance on the provided
    Pacman engine.
    """

    def __init__(self, dealer, layout_representation, use_random_ghost=False):
        """Initialize an interface to the Pacman game simulator.

        :param layout_representation: either the layout filename (located in layouts/) or an actual layout object.
        :param use_random_ghost: whether to use the random or the directional ghost agent.
        """
        self.dealer = dealer

        if isinstance(layout_representation, str):
            self.layout = layout.getLayout(layout_representation)
        else:  # we've been directly given a layout
            self.layout = layout_representation
        self.my_name = "Pacman"

        if self.dealer.use_graphics:
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

    def set_agent(self, agent):
        """Sets Pacman's agent."""
        self.pacman_agent = Agent(agent, self)
        self.reinitialize()

    def clone(self):
        new_sim = PacmanState(self.dealer, self.layout)
        new_sim.current_state = self.current_state
        return new_sim

    def number_of_players(self):
        return self.num_players

    def set(self, sim):
        self.current_state = sim.current_state

    def is_terminal(self):
        return self.current_state.isWin() or self.current_state.isLose()

    def process(self, state, game_object):
        """Wrapper to help with ending the game."""
        if state.isWin():
            self.final_score = state.data.score
            self.won = True
            pacman.ClassicGameRules.win(self, state=state, game=game_object)
        if state.isLose():
            self.final_score = state.data.score
            pacman.ClassicGameRules.lose(self, state=state, game=game_object)

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
