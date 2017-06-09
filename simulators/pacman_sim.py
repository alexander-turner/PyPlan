from abstract import absstate
import sys
sys.path.append('C:\\Users\\Alex\\OneDrive\\Documents\\Classes\\OSU\\Research\\PyPlan\\simulators\\pacmancode')
import pacman
import game
import layout
import textDisplay
import ghostAgents
import copy

class PacmanStateClass(pacman.GameState):

    def __init__(self, layoutName, agent, silent=False):
        self.layoutName = layoutName
        self.layout = layout.getLayout(self.layoutName)
        if silent:  # TODO: Add this feature
            self.display = 0
        else:
            self.display = textDisplay.PacmanGraphics()

        self.ghostAgents = [ghostAgents.DirectionalGhost(i) for i in range(1, self.layout.getNumGhosts()+1)]
        self.agent = PolicyAgent(agent, self.ghostAgents)

        self.game = pacman.ClassicGameRules.newGame(self, layout=self.layout, pacmanAgent=self.agent, ghostAgents=self.ghostAgents, display=self.display)
        self.current_state = self.game.state
        self.num_players = self.game.state.getNumAgents()

    def clone(self):
        return PacmanStateClass(self.layoutName, self.agent.policy)

    def number_of_players(self):
        return self.num_players

    def set(self, state):
        self.current_state = state.clone()

    # Reinitialize using the given layout, Pacman agent, and ghost agents.
    def initialize(self):
        self.game = pacman.ClassicGameRules.newGame(self.layout, self.agent, self.ghostAgents, self.display)

    def current_game_outcome(self):
        if self.game.isWin():
            return 1
        elif self.game.isLose():
            return -1
        else:
            return 0

    # Wrapper to help with ending the game
    def process(self, state, game):
        if state.isWin():
            pacman.ClassicGameRules.win(self, state=state, game=game)
        if state.isLose():
            pacman.ClassicGameRules.lose(self, state=state, game=game)


class PolicyAgent(game.Agent):
    def __init__(self, policy, ghosts, index=0, current_state=0):
        self.policy = policy
        self.ghosts = ghosts
        self.index = index
        self.current_state = current_state

    def getAction(self, state):
        # Define functions so the policy can interface with Pacman
        self.applyFunctionsToState(state)
        self.current_state = state
        return self.policy.select_action(state)

    def returnIndex(self):
        return self.index + 1 # accounts for our different indexing of agents

    def applyFunctionsToState(self, state):
        state.get_action_list = state.getLegalActions
        state.get_actions = state.getLegalActions

        def clone():
            return copy.deepcopy(state)
        state.clone = clone

        state.number_of_players = state.getNumAgents
        state.get_current_player = self.returnIndex
        state.take_action = self.take_action

        def is_terminal():
            return state.isWin() or state.isLose()
        state.is_terminal = is_terminal

    # Take the given action and update the state accordingly
    def take_action(self, action):
        state = self.current_state
        if action in state.getLegalActions():  # Bandit algorithms try each arm - sometimes illegal
            new_state = state.generateSuccessor(state.get_current_player()-1, action)
            for ghostInd, ghost in enumerate(self.ghosts): #simulate ghost movements
                if new_state.isWin() or new_state.isLose():
                    break
                ghostAction = ghost.getAction(state)
                new_state = new_state.generateSuccessor(ghostInd+1, ghostAction)
            self.applyFunctionsToState(new_state)
            reward = new_state.getScore() - state.getScore()  # reward pacman gets
            rewards = [-1*reward]*state.number_of_players()  # reward ghosts get
            rewards[0] *= -1  # correct pacman reward
            self.current_state = new_state  # not really updating current state?
            return rewards  # how much our score increased because of this action
