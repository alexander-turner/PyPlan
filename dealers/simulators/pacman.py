import sys
import os
sys.path.append(os.path.abspath('..\\dealers\\simulators\\pacmancode'))  # assume we're running from demos/
from abstract import abstract_state
from dealers.simulators.pacmancode import *


class PacmanState(abstract_state.AbstractState):
    """An interface to run bandit algorithms on the Pacman simulator provided by Berkeley.

    Give multiple agents to compare results.

    The simulator can be found at http://ai.berkeley.edu/project_overview.html.

    The simulator processes eating pellets before ghost detection, so Pacman may make illegal-looking moves - this is
     simply how Berkeley's underlying game rules work (and not a result of this simulator's implementation).
    """
    current_player = 0
    env_name = "Pacman"

    def __init__(self, dealer, layout_repr, use_random_ghost=False):
        """Initialize an interface to the Pacman game simulator.

        :param layout_repr: either the layout filename (located in layouts/) or an actual layout object.
        :param use_random_ghost: whether to use the random or the directional ghost agent.
        """
        self.dealer = dealer  # pointer to parent dealer

        self.layout = layout.getLayout(layout_repr) if isinstance(layout_repr, str) else layout_repr
        self.display = graphicsDisplay.PacmanGraphics() if self.dealer.use_graphics else textDisplay.PacmanGraphics()
        self.show_moves = True

        self.pacman_agent = None
        ghost_agent = ghostAgents.RandomGhost if use_random_ghost else ghostAgents.DirectionalGhost
        self.ghost_agents = [ghost_agent(i) for i in range(1, self.layout.getNumGhosts() + 1)]

        self.game = pacman.ClassicGameRules.newGame(self, layout=self.layout, pacmanAgent=self.pacman_agent,
                                                    ghostAgents=self.ghost_agents, display=self.display, quiet=True)
        self.current_state = self.game.state
        self.num_players = self.game.state.getNumAgents()

        self.won = False
        self.time_step_count = 0

    def reinitialize(self):
        """Reinitialize using the defined layout, Pacman agent, ghost agents, and display."""
        self.game = pacman.ClassicGameRules.newGame(self, layout=self.layout, pacmanAgent=self.pacman_agent,
                                                    ghostAgents=self.ghost_agents, display=self.display, quiet=True)
        self.current_state = self.game.state
        self.won = False
        self.time_step_count = 0  # how many total turns have elapsed

    def set_agent(self, agent):
        """Sets Pacman's agent."""
        self.pacman_agent = Agent(agent, self)
        self.reinitialize()

    def clone(self):
        new_state = PacmanState(self.dealer, self.layout)
        new_state.current_state = self.current_state
        return new_state

    def set(self, sim):
        self.current_state = sim.current_state

    def take_action(self, action):
        """Take the action and update the current state accordingly."""
        new_state = self.current_state.generateSuccessor(self.current_player, action)  # simulate Pacman movement

        for ghost_idx, ghost in enumerate(self.ghost_agents):  # simulate ghost movements
            if new_state.isWin() or new_state.isLose():
                break
            ghost_action = ghost.getAction(new_state)
            new_state = new_state.generateSuccessor(ghost_idx + 1, ghost_action)

        reward = new_state.getScore() - self.current_state.getScore()  # reward Pacman gets
        rewards = [reward if player_idx == 0 else -1 * reward for player_idx in range(self.num_players)]

        self.current_state = new_state

        self.time_step_count += 1  # mark that another time step has elapsed

        return rewards  # how much our score increased because of this action

    def get_actions(self):
        return self.current_state.getLegalActions()

    def get_value_bounds(self):
        return {'defeat': -500, 'victory': 500,
                'min non-terminal': -1, 'max non-terminal': 200,
                'pre-computed min': None, 'pre-computed max': None,
                'evaluation function': None}

    def is_terminal(self):
        return self.current_state.isWin() or self.current_state.isLose()

    def process(self, state, game_object):
        """Wrapper to help with ending the game."""
        if state.isWin():
            self.won = True
            pacman.ClassicGameRules.win(self, state=state, game=game_object)
        if state.isLose():
            pacman.ClassicGameRules.lose(self, state=state, game=game_object)

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return self.current_state.__hash__()


class Agent(game.Agent):
    """A wrapper to let the policy interface with the Pacman engine."""

    def __init__(self, policy, parent):
        self.policy = policy
        self.parent = parent  # pointer to parent PacmanState structure

    def getAction(self, state):
        self.parent.current_state = state
        self.parent.time_step_count += 1
        return self.policy.select_action(self.parent)
