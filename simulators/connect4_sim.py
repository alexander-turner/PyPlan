import copy
from abstract import absstate


class Connect4StateClass(absstate.AbstractState):
    num_players = 2
    board_height = 6
    board_width = 7

    def __init__(self):
        self.state_val = [0, 0]
        self.current_player = 0
        self.game_outcome = None  # 0 - player1 is winner, 1 - player2 is winner, None - no winner

    def clone(self):
        new_state = copy.deepcopy(self)
        return new_state

    def number_of_players(self):
        return Connect4StateClass.num_players

    def set(self, state):
        self.state_val[0] = state.state_val[0]
        self.state_val[1] = state.state_val[1]
        self.current_player = state.current_player
        self.game_outcome = state.game_outcome

    def initialize(self):
        self.state_val = [0, 0]
        self.current_player = 0
        self.game_outcome = None  # 0 - player1 is winner, 1 - player2 is winner, None - no winner

    def take_action(self, action):
        position = action[0]
        value = action[1]

        self.state_val[value - 1] |= 1 << position
        self.game_outcome = self.current_game_outcome()

        self.current_player = (self.current_player+1) % self.num_players  # change turn

        if self.game_outcome == 0:
            return [1.0, -1.0]
        elif self.game_outcome == 1:
            return [-1.0, 1.0]
        else:
            return [0.0, 0.0]

    def get_actions(self):
        actions_list = []
        current_board = 0

        for player_board in range(Connect4StateClass.num_players):
            current_board |= self.state_val[player_board]

        board_size = ((Connect4StateClass.board_height + 1) * Connect4StateClass.board_width)
        current_board = bin(current_board)[2:].zfill(board_size)[::-1]
        for column in range(Connect4StateClass.board_width):
            curr_val = (Connect4StateClass.board_height + ((Connect4StateClass.board_height + 1) * column))
            curr_val -= 1
            if int(current_board[curr_val]) == 0:
                while curr_val >= (Connect4StateClass.board_height + 1) * column:
                    if int(current_board[curr_val]) == 1:
                        actions_list.append(action)
                        break
                    else:
                        action = [curr_val, self.current_player]

                    if curr_val == Connect4StateClass.board_width * column:
                        actions_list.append(action)

                    curr_val -= 1

        return actions_list

    def is_terminal(self):
        return self.game_outcome is not None

    def get_current_player(self):
        return self.current_player

    def current_game_outcome(self):
        for player in range(Connect4StateClass.num_players):
            curr_board = self.state_val[player]
            temp = bin(curr_board)

            # LEFT DIAGONAL
            transform = curr_board & (curr_board >> Connect4StateClass.board_height)
            if transform & (transform >> (2 * Connect4StateClass.board_height)):
                return player

            # RIGHT DIAGONAL
            transform = curr_board & (curr_board >> (Connect4StateClass.board_width + 1))
            if transform & (transform >> (2 * (Connect4StateClass.board_width + 1))):
                return player

            # HORIZONTAL
            transform = curr_board & (curr_board >> Connect4StateClass.board_width)
            if transform & (transform >> (2 * Connect4StateClass.board_width)):
                return player

            # VERTICAL
            transform = curr_board & (curr_board >> 1)
            if transform & (transform >> 2):
                return player

        # NO WINS BUT CHECK FOR DRAW
        current_board = 0

        for player_board in range(Connect4StateClass.num_players):
            current_board |= self.state_val[player_board]

        board_size = (
                     Connect4StateClass.board_height * Connect4StateClass.board_width) + Connect4StateClass.board_height
        current_board = bin(current_board)[2:].zfill(board_size)[::-1]
        excluded_vals = [(Connect4StateClass.board_height + (Connect4StateClass.board_height + 1) * x) for x in
                         range(Connect4StateClass.board_width)]

        for val in range(board_size):
            if val not in excluded_vals:
                if int(current_board[val]) == 0:
                    return None
        return 0

    def __eq__(self, other):
        return self.state_val == other.state_val

    def __hash__(self):
        return hash(tuple(self.state_val))

    def __repr__(self):
        curr_row = 6
        output = ""
        while curr_row >= 0:
            curr_col = 0
            while curr_col < 7:
                board_size = ((6 + 1) * 7)
                player_num = 1
                is_printed = False
                for player_board in self.state_val:
                    curr_board = bin(player_board)[2:].zfill(board_size)[::-1]
                    if int(curr_board[curr_row + ((6 + 1) * curr_col)]) == 1:
                        output += str(player_num) + ""
                        is_printed = True
                    player_num += 1

                if is_printed is False:
                    output += "0"

                curr_col += 1

            output += "\n"

            curr_row -= 1

        return output
