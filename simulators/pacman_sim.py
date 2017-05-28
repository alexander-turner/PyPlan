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

    def __init__(self, layoutName, agent):
        self.state_val = [0, 0]
        self.game_outcome = None

        self.layoutName = layoutName
        self.layout = layout.getLayout(self.layoutName)
        self.display = textDisplay.PacmanGraphics()

        self.current_player = 0 # pacman index - CHECK
        self.agent = PolicyAgent(agent)
        self.ghostAgents = [ghostAgents.DirectionalGhost(i) for i in range(1, self.layout.getNumGhosts()+1)]

        self.game = pacman.ClassicGameRules.newGame(self, layout=self.layout, pacmanAgent=self.agent, ghostAgents=self.ghostAgents, display=self.display)
        self.num_players = self.game.state.getNumAgents()

    def clone(self):
        return PacmanStateClass(self.layoutName, self.agent.policy)

    def number_of_players(self):
        return self.num_players

    def set(self, state):
        self.state_val[0] = state.state_val[0]
        self.state_val[1] = state.state_val[1]
        self.current_player = state.current_player
        self.game_outcome = state.game_outcome

    def initialize(self): # how to initialize multiple ghosts?
        self.state_val = [0, 0]
        self.current_player = 0 # pacman index - CHECK
        self.game = pacman.ClassicGameRules.newGame(self.layout, self.agent, self.ghostAgents, 1)
        self.game_outcome = None


    def get_actions(self):
        return self.game.state.getLegalActions(self.game.state)

    def is_terminal(self):
        return self.game_outcome is not None

    def get_current_player(self):
        return self.current_player

    def current_game_outcome(self):
        if self.game.isWin():
            return 1
        elif self.game.isLose():
            return -1
        else:
            return 0

    def process(self, state, game):
        """
        Checks to see whether it is time to end the game.
        """
        if state.isWin(): pacman.ClassicGameRules.win(self, state=state, game=game)
        if state.isLose(): pacman.ClassicGameRules.lose(self, state=state, game=game)

    def __eq__(self, other):
        return hasattr(other, 'data') and self.data == other.data

    def __hash__(self):
        return hash(self.data)

    def __repr__(self):
        output = ""
        return output


class PolicyAgent(game.Agent):
    def __init__(self, policy, index=0):
        self.index = index
        self.policy = policy

    def getAction(self, state):
        state.get_action_list = state.getLegalActions  # Define so the policy can interface with pacman
        state.get_actions = state.getLegalActions

        def clone():
            return copy.deepcopy(state)
        state.clone = clone

        state.number_of_players = state.getNumAgents
        state.get_current_player = self.returnIndex

        def take_action(action):
            new_state = state.generateSuccessor(state.get_current_player()-1, action)
            return [new_state.getScore() - state.getScore()]  # how much our score increased because of this action
        state.take_action = take_action

        def isTerminal():
            return state.isWin() or state.isLose()
        state.is_terminal = isTerminal

        return self.policy.select_action(state)

    def returnIndex(self):
        return self.index + 1 # accounts for our different indexing of agents

