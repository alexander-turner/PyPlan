import copy
from abstract import absstate


class Connect4State(absstate.AbstractState):
    def __init__(self):
        self.state_val = [0, 0]

        self.board_height = 6
        self.board_width = 7

        self.num_players = 2
        self.current_player = 0

        self.game_outcome = None  # 0 - player1 is winner, 1 - player2 is winner, None - no winner
        self.my_name = "Connect 4"

    def clone(self):
        new_state = copy.deepcopy(self)
        return new_state

    def number_of_players(self):
        return self.num_players

    def set(self, state):
        self.state_val[0] = state.state_val[0]
        self.state_val[1] = state.state_val[1]
        self.current_player = state.current_player
        self.game_outcome = state.game_outcome

    def reinitialize(self):
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

        for player_board in range(self.num_players):
            current_board |= self.state_val[player_board]

        board_size = ((self.board_height + 1) * self.board_width)
        current_board = bin(current_board)[2:].zfill(board_size)[::-1]
        for column in range(self.board_width):
            curr_val = (self.board_height + ((self.board_height + 1) * column))
            curr_val -= 1
            if int(current_board[curr_val]) == 0:
                while curr_val >= (self.board_height + 1) * column:
                    if int(current_board[curr_val]) == 1:
                        actions_list.append(action)
                        break
                    else:
                        action = [curr_val, self.current_player]

                    if curr_val == self.board_width * column:
                        actions_list.append(action)

                    curr_val -= 1

        return actions_list

    def is_terminal(self):
        return self.game_outcome is not None

    def get_current_player(self):
        return self.current_player

    def set_current_player(self, player_index):
        self.current_player = player_index

    def current_game_outcome(self):
        for player in range(self.num_players):
            curr_board = self.state_val[player]

            # Left diagonal
            transform = curr_board & (curr_board >> self.board_height)
            if transform & (transform >> (2 * self.board_height)):
                return player

            # Right diagonal
            transform = curr_board & (curr_board >> (self.board_width + 1))
            if transform & (transform >> (2 * (self.board_width + 1))):
                return player

            # Horizontal
            transform = curr_board & (curr_board >> self.board_width)
            if transform & (transform >> (2 * self.board_width)):
                return player

            # Vertical
            transform = curr_board & (curr_board >> 1)
            if transform & (transform >> 2):
                return player

        # Check for draw
        current_board = 0

        for player_board in range(self.num_players):
            current_board |= self.state_val[player_board]

        board_size = self.board_height * (self.board_width + 1)
        current_board = bin(current_board)[2:].zfill(board_size)[::-1]
        excluded_vals = [(self.board_height + (self.board_height + 1) * x) for x in
                         range(self.board_width)]

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
        output = ""
        board_size = (self.board_height + 1) * self.board_width
        for row in range(self.board_height, -1, -1):
            for col in range(self.board_width):
                player_num = 1
                is_printed = False
                for player_board in self.state_val:
                    curr_board = bin(player_board)[2:].zfill(board_size)[::-1]
                    if int(curr_board[row + ((self.board_height + 1) * col)]) == 1:
                        output += str(player_num) + ""
                        is_printed = True
                    player_num += 1

                if is_printed is False:
                    output += "0"
            output += "\n"

        return output
