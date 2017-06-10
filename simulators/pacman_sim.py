import copy
import sys
import os
sys.path.append(os.path.abspath('simulators\\pacmancode'))
import pacman
import game
import layout
import textDisplay
import graphicsDisplay
import ghostAgents

"""
An interface to run bandit algorithms on the Pacman simulator provided by Berkeley. 
The simulator can be found at http://ai.berkeley.edu/project_overview.html.
"""
class PacmanStateClass(pacman.GameState):

    def __init__(self, layoutName, agent, useGraphics=True):
        self.layoutName = layoutName
        self.layout = layout.getLayout(self.layoutName)

        if useGraphics:
            self.display = graphicsDisplay.PacmanGraphics()
        else:
            self.display = textDisplay.PacmanGraphics()

        self.ghostAgents = [ghostAgents.DirectionalGhost(i) for i in range(1, self.layout.getNumGhosts()+1)]
        self.agent = PolicyAgent(agent, self)

        self.game = pacman.ClassicGameRules.newGame(self, layout=self.layout, pacmanAgent=self.agent,
                                                    ghostAgents=self.ghostAgents, display=self.display)
        self.current_state = self.game.state
        self.applyFunctionsToState(self.current_state)
        self.num_players = self.game.state.getNumAgents()
        self.current_player = 0

    # Reinitialize using the defined layout, Pacman agent, ghost agents, and display.
    def initialize(self):
        self.game = pacman.ClassicGameRules.newGame(self.layout, self.agent, self.ghostAgents, self.display)
        self.current_state = self.game.state
        self.applyFunctionsToState(self.current_state)
        self.num_players = self.game.state.getNumAgents()
        self.current_player = 0

    def clone(self):
        return copy.deepcopy(self)

    def number_of_players(self):
        return self.num_players

    def set(self, state):
        self.current_state = state.clone()

    def is_terminal(self):
        return self.current_state.isWin() or self.current_state.isLose()

    # Wrapper to help with ending the game
    def process(self, state, game):
        if state.isWin():
            pacman.ClassicGameRules.win(self, state=state, game=game)
        if state.isLose():
            pacman.ClassicGameRules.lose(self, state=state, game=game)

    # Return index of current player
    def returnIndex(self):
        return self.current_player + 1  # accounts for our different indexing of agents

    def get_current_state(self):
        return self.current_state

    # Take the given action and update the state accordingly
    def take_action(self, action):
        state = self.current_state

        new_state = state.generateSuccessor(state.get_current_player() - 1,
                                            action)  # -1 since different indexing systems
        for ghostInd, ghost in enumerate(self.ghostAgents):  # simulate ghost movements
            if new_state.isWin() or new_state.isLose():
                break
            ghostAction = ghost.getAction(state)
            new_state = new_state.generateSuccessor(ghostInd + 1, ghostAction)

        reward = new_state.getScore() - state.getScore()  # reward Pacman gets
        rewards = [-1 * reward] * state.number_of_players()  # reward ghosts get
        rewards[0] *= -1  # correct Pacman reward

        self.applyFunctionsToState(new_state)
        self.current_state = new_state  # not really updating current state?
        return rewards  # how much our score increased because of this action

    # Define functions so the policy can interface with Pacman
    def applyFunctionsToState(self, state):
        def clone():
            return copy.deepcopy(state)
        state.clone = clone
        state.set = self.set

        state.number_of_players = state.getNumAgents
        state.get_current_player = self.returnIndex
        state.get_current_state = self.get_current_state
        state.get_actions = state.getLegalActions
        state.take_action = self.take_action

        state.is_terminal = self.is_terminal


class PolicyAgent(game.Agent):
    def __init__(self, policy, pacstate):
        self.policy = policy
        self.pacstate = pacstate  # pointer to parent pacstate structure

    def getAction(self, state):
        self.pacstate.applyFunctionsToState(state=state)
        self.pacstate.current_state = state
        return self.policy.select_action(state)
