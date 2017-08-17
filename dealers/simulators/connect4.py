import copy
from abstract import abstract_state


class Connect4State(abstract_state.AbstractState):
    env_name = "Connect 4"
    num_players = 2
    height, width = 6, 7

    def __init__(self):
        self.state_val = [0, 0]
        self.current_player = 0
        self.game_outcome = None  # 0 - player1 is winner, 1 - player2 is winner, None - no winner

    def reinitialize(self):
        self.state_val = [0, 0]
        self.current_player = 0
        self.game_outcome = None  # 0 - player1 is winner, 1 - player2 is winner, None - no winner

    def clone(self):
        new_state = copy.deepcopy(self)
        return new_state

    def set(self, state):
        self.state_val = copy.deepcopy(state.state_val)
        self.current_player = state.current_player
        self.game_outcome = state.game_outcome

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

        board_size = ((self.height + 1) * self.width)
        current_board = bin(current_board)[2:].zfill(board_size)[::-1]
        for column in range(self.width):
            curr_val = (self.height + ((self.height + 1) * column))
            curr_val -= 1
            if int(current_board[curr_val]) == 0:
                while curr_val >= (self.height + 1) * column:
                    if int(current_board[curr_val]) == 1:
                        actions_list.append(action)
                        break
                    else:
                        action = tuple([curr_val, self.current_player])

                    if curr_val == self.width * column:
                        actions_list.append(action)

                    curr_val -= 1

        return actions_list

    def get_value_bounds(self):
        return {'defeat': -1, 'victory': 1,
                'min non-terminal': 0, 'max non-terminal': 0,
                'pre-computed min': -1, 'pre-computed max': 1,
                'evaluation function': None}

    def is_terminal(self):
        return self.game_outcome is not None

    def current_game_outcome(self):
        for player in range(self.num_players):
            curr_board = self.state_val[player]

            # Left diagonal
            transform = curr_board & (curr_board >> self.height)
            if transform & (transform >> (2 * self.height)):
                return player

            # Right diagonal
            transform = curr_board & (curr_board >> (self.width + 1))
            if transform & (transform >> (2 * (self.width + 1))):
                return player

            # Horizontal
            transform = curr_board & (curr_board >> self.width)
            if transform & (transform >> (2 * self.width)):
                return player

            # Vertical
            transform = curr_board & (curr_board >> 1)
            if transform & (transform >> 2):
                return player

        # Check for draw
        current_board = 0

        for player_board in range(self.num_players):
            current_board |= self.state_val[player_board]

        board_size = self.height * (self.width + 1)
        current_board = bin(current_board)[2:].zfill(board_size)[::-1]
        excluded_vals = [(self.height + (self.height + 1) * x) for x in
                         range(self.width)]

        for val in range(board_size):
            if val not in excluded_vals:
                if int(current_board[val]) == 0:
                    return None
        return 0

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(tuple(self.state_val + [self.current_player]))

    def __str__(self):
        output = ""
        board_size = (self.height + 1) * self.width
        for row in range(self.height, -1, -1):
            for col in range(self.width):
                player_num = 1
                is_printed = False
                for player_board in self.state_val:
                    curr_board = bin(player_board)[2:].zfill(board_size)[::-1]
                    if int(curr_board[row + ((self.height + 1) * col)]) == 1:
                        output += str(player_num) + ""
                        is_printed = True
                    player_num += 1

                if is_printed is False:
                    output += "0"
            output += "\n"

        return output
