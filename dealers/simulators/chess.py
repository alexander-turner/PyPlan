import copy
from abstract import abstract_state
from dealers.simulators.chesscode import chess


class ChessState(abstract_state.AbstractState):
    env_name = "Chess"
    num_players = 2

    def __init__(self):
        self.current_state = chess.Board()
        self.current_player = 0
        self.game_outcome = None  # 0 - player1 is winner, 'draw' - draw, 1 - player2 is winner, None - game not over

    def reinitialize(self):
        self.current_state = chess.Board()
        self.current_player = 0
        self.game_outcome = None

    def clone(self):
        new_state = copy.deepcopy(self)
        return new_state

    def set(self, state):
        self.current_state = copy.deepcopy(state.current_state)
        self.current_player = state.current_player
        self.game_outcome = state.game_outcome

    def take_action(self, action):
        self.current_state.update_board(action)
        action[0].has_moved = True  # mark that the piece has been moved (for castling purposes)
        self.current_player = (self.current_player + 1) % self.num_players

        current_player = self.current_state.players[self.get_current_color()]
        actions = self.get_actions()

        if len(actions) == 0:
            if current_player.in_check:  # current player lost
                self.game_outcome = (self.current_player + 1) % self.num_players  # other player won
                rewards = [1 for _ in range(self.num_players)]
                rewards[self.current_player] *= -1
                return rewards
            else:  # draw
                self.game_outcome = 'draw'
        # todo check other player too?
        return 0, 0

    def get_actions(self):
        return self.current_state.players[self.get_current_color()].get_moves()

    def number_of_players(self):
        return self.num_players

    def get_current_player(self):
        return self.current_player

    def get_current_color(self):
        return 'white' if self.current_player == 0 else 'black'

    def set_current_player(self, player_index):
        self.current_player = player_index

    def get_value_bounds(self):
        return {'defeat': -1, 'victory': 1,
                'min non-terminal': 0, 'max non-terminal': 0,
                'pre-computed min': -1, 'pre-computed max': 1,
                'evaluation function': None}

    def is_terminal(self):
        return self.game_outcome is not None

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash((self.current_state, self.current_player))

    def __str__(self):
        return self.current_state.__str__()  # print board
