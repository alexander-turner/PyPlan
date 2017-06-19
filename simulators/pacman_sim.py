import sys
import os
sys.path.append(os.path.abspath('simulators\\pacmancode'))
import pacman
import game
import layout
import textDisplay
import graphicsDisplay
import ghostAgents


class PacmanStateClass(pacman.GameState):
    """An interface to run bandit algorithms on the Pacman simulator provided by Berkeley.

    The simulator can be found at http://ai.berkeley.edu/project_overview.html.

    Note that unlike other interfaces, this is not compatible with the dealer simulator due its reliance on the provided
    Pacman engine.
    """

    def __init__(self, layout_name, agent_construct, use_random_ghost=False, use_graphics=True):
        """Initialize an interface to the Pacman game simulator.

        :param layout_name: the filename of the layout (located in layouts/).
        :param agent_construct: the bandit to be used for Pacman.
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

        self.agent_construct = agent_construct
        self.agent = PolicyAgent(agent_construct, self)

        self.game = pacman.ClassicGameRules.newGame(self, layout=self.layout, pacmanAgent=self.agent,
                                                    ghostAgents=self.ghost_agents, display=self.display)
        self.current_state = self.game.state
        self.num_players = self.game.state.getNumAgents()
        self.current_player = 0

    def initialize(self):
        """Reinitialize using the defined layout, Pacman agent, ghost agents, and display."""
        self.game = pacman.ClassicGameRules.newGame(self.layout, self.agent, self.ghost_agents, self.display)
        self.current_state = self.game.state
        self.num_players = self.game.state.getNumAgents()
        self.current_player = 0

    def run(self):
        self.game.run()

    def clone(self):
        new_sim_obj = PacmanStateClass(self.layoutName, self.agent_construct)
        new_sim_obj.current_state = self.current_state
        new_sim_obj.current_player = self.current_player
        return new_sim_obj

    def number_of_players(self):
        return self.num_players

    def set(self, sim):
        self.current_state = sim.current_state

    def is_terminal(self):
        return self.current_state.isWin() or self.current_state.isLose()

    def process(self, state, game):
        """Wrapper to help with ending the game."""
        if state.isWin():
            pacman.ClassicGameRules.win(self, state=state, game=game)
        if state.isLose():
            pacman.ClassicGameRules.lose(self, state=state, game=game)

    def get_current_player(self):
        """Returns one-indexed index of current player (for compatibility with existing bandit library)."""
        return self.current_player + 1

    def get_current_state(self):
        return self.current_state

    def take_action(self, action):
        """Take the action and update the current state accordingly."""
        new_state = self.current_state.generateSuccessor(self.get_current_player() - 1,
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

    def __hash__(self):
        return self.current_state.__hash__()


class PolicyAgent(game.Agent):
    """A wrapper to let the policy interface with the Pacman engine."""
    def __init__(self, policy, pac_state):
        self.policy = policy
        self.pac_state = pac_state  # pointer to parent PacmanStateClass structure

    def getAction(self, state):
        self.pac_state.current_state = state
        return self.policy.select_action(self.pac_state)
