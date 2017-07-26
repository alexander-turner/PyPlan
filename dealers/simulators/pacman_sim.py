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
                                                    ghostAgents=self.ghost_agents, display=self.display, quiet=True)
        self.current_state = self.game.state
        self.num_players = self.game.state.getNumAgents()

        self.final_score = float("-inf")  # we have yet to obtain a final score
        self.won = False
        self.time_step_count = 0

    def reinitialize(self):
        """Reinitialize using the defined layout, Pacman agent, ghost agents, and display."""
        self.game = pacman.ClassicGameRules.newGame(self, layout=self.layout, pacmanAgent=self.pacman_agent,
                                                    ghostAgents=self.ghost_agents, display=self.display, quiet=True)
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

    def set(self, sim):
        self.current_state = sim.current_state

    def take_action(self, action):
        """Take the action and update the current state accordingly."""
        new_state = self.current_state.generateSuccessor(self.get_current_player(), action)  # simulate Pacman movement

        for ghost_idx, ghost in enumerate(self.ghost_agents):  # simulate ghost movements
            if new_state.isWin() or new_state.isLose():
                break
            ghost_action = ghost.getAction(new_state)
            new_state = new_state.generateSuccessor(ghost_idx + 1, ghost_action)

        reward = new_state.getScore() - self.current_state.getScore()  # reward Pacman gets
        rewards = [-1 * reward] * self.number_of_players()  # reward ghosts get
        rewards[0] *= -1  # correct Pacman reward

        self.current_state = new_state

        self.time_step_count += 1  # mark that another time step has elapsed

        return rewards  # how much our score increased because of this action

    def get_actions(self):
        return self.current_state.getLegalActions()

    def number_of_players(self):
        return self.num_players

    def get_current_player(self):
        """Pacman is the only player."""
        return 0

    def get_value_bounds(self):
        return {'defeat': -500, 'victory': 500,
                'min non-terminal': -1, 'max non-terminal': 200,
                'pre-computed min': None, 'pre-computed max': None}

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

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return self.current_state.__hash__()


class Agent(game.Agent):
    """A wrapper to let the policy interface with the Pacman engine."""

    def __init__(self, policy, pac_state):
        self.policy = policy
        self.pac_state = pac_state  # pointer to parent PacmanState structure

    def getAction(self, state):
        self.pac_state.current_state = state
        self.pac_state.time_step_count += 1
        return self.policy.select_action(self.pac_state)
