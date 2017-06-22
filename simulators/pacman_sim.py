import sys
import os
sys.path.append(os.path.abspath('simulators\\pacmancode'))
from abstract import absstate
import tabulate  # install tabulate in the new environment
import numpy
import pacman
import game
import layout
import textDisplay
import graphicsDisplay
import ghostAgents


class PacmanStateClass(absstate.AbstractState):
    """An interface to run bandit algorithms on the Pacman simulator provided by Berkeley.

    Give multiple agents to compare results.

    The simulator can be found at http://ai.berkeley.edu/project_overview.html.

    Note that unlike other interfaces, this is not compatible with the dealer simulator due its reliance on the provided
    Pacman engine.
    """

    # TODO: Add a way to run x trials. Compare between policies.
    def __init__(self, layout_name, agents, use_random_ghost=False, use_graphics=True):
        """Initialize an interface to the Pacman game simulator.

        :param layout_name: the filename of the layout (located in layouts/).
        :param agents: list of the bandit(s) to be used to control Pacman. If multiple, first will be used.
        :param use_random_ghost: whether to use the random or the directional ghost agent.
        :param use_graphics: whether to use the graphics or the text display.
        """
        self.layoutName = layout_name
        self.layout = layout.getLayout(self.layoutName)

        if use_graphics:
            self.display = graphicsDisplay.PacmanGraphics()
        else:
            self.display = textDisplay.PacmanGraphics()

        if use_random_ghost:
            self.ghost_agents = [ghostAgents.RandomGhost(i) for i in range(1, self.layout.getNumGhosts() + 1)]
        else:
            self.ghost_agents = [ghostAgents.DirectionalGhost(i) for i in range(1, self.layout.getNumGhosts() + 1)]

        self.agents = agents
        self.current_agent, self.current_agent_idx = self.agents[0], 0  # Take first agent from the list
        self.pacman_agent = Agent(self.current_agent, self)

        self.game = pacman.ClassicGameRules.newGame(self, layout=self.layout, pacmanAgent=self.pacman_agent,
                                                    ghostAgents=self.ghost_agents, display=self.display)
        self.current_state = self.game.state
        self.final_score = float("-inf")  # demarcates we have yet to obtain a final score
        self.won = False  # True := win, False := loss
        self.num_players = self.game.state.getNumAgents()

    def initialize(self):
        """Reinitialize using the defined layout, Pacman agent, ghost agents, and display."""
        self.game = pacman.ClassicGameRules.newGame(self, layout=self.layout, pacmanAgent=self.pacman_agent,
                                                    ghostAgents=self.ghost_agents, display=self.display)
        self.current_state = self.game.state
        self.final_score = float("-inf")
        self.won = False

    def run(self, num_trials=1, verbose=False):
        """Runs num_trials trials for each of the provided agents, neatly displaying results (if requested)."""
        table = []
        headers = ["Agent Name", "Average Score", "Win Percentage"]
        for _ in self.agents:
            rewards = [0] * num_trials
            wins = 0
            for i in range(num_trials):
                self.initialize()  # reset the game
                self.game.run()
                rewards[i] = self.final_score
                if self.won:
                    wins += 1
            table.append([self.current_agent.agentname, numpy.mean(rewards), wins / num_trials])
            self.load_next_agent()
        if verbose:
            print(tabulate(table, headers))

    def load_next_agent(self):
        """Generates the next agent from the provided list of agents, resetting to the start if necessary."""
        self.current_agent_idx = (self.current_agent_idx + 1) % len(self.agents)
        self.current_agent = self.agents[self.current_agent_idx]
        self.pacman_agent = Agent(self.current_agent, self)

    def clone(self):
        new_sim = PacmanStateClass(self.layoutName, self.current_agent)
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
        new_state = self.current_state.generateSuccessor(self.get_current_player(),
                                                         action)  # -1 since different indexing systems
        for ghostInd, ghost in enumerate(self.ghost_agents):  # simulate ghost movements
            if new_state.isWin() or new_state.isLose():
                break
            ghost_action = ghost.getAction(new_state)
            new_state = new_state.generateSuccessor(ghostInd + 1, ghost_action)

        reward = new_state.getScore() - self.current_state.getScore()  # reward Pacman gets
        rewards = [-1 * reward] * self.number_of_players()  # reward ghosts get
        rewards[0] *= -1  # correct Pacman reward

        self.current_state = new_state
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
        return self.policy.select_action(self.pac_state)

