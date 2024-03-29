import copy
import numpy as np
from abstract import abstract_state


class OthelloState(abstract_state.AbstractState):
    env_name = "Othello"
    num_players = 2
    state_val = [[0] * 8 for _ in range(8)]
    state_val[3][3] = 2
    state_val[3][4] = 1
    state_val[4][3] = 1
    state_val[4][4] = 2
    original_state = state_val

    def __init__(self):
        self.current_state = copy.deepcopy(self.original_state)
        self.game_outcome = None
        self.game_over = False

    def reinitialize(self):
        self.current_state = copy.deepcopy(self.original_state)
        self.current_player = 0
        self.game_outcome = None
        self.game_over = False

    def clone(self):
        return copy.deepcopy(self)

    def set(self, sim):
        self.current_state = copy.deepcopy(sim.current_state)
        self.game_outcome = sim.game_outcome
        self.game_over = sim.game_over

    def take_action(self, action):
        position = action['position']
        value = action['value']

        # Check for null action
        if value == -1:
            return np.array([0.0] * self.num_players)

        self.current_state[position[0]][position[1]] = value

        # Update the board
        i = position[0]
        j = position[1]
        self.color_coins(value, [i - 1, j], "U", True)
        self.color_coins(value, [i + 1, j], "D", True)
        self.color_coins(value, [i, j + 1], "R", True)
        self.color_coins(value, [i, j - 1], "L", True)
        self.color_coins(value, [i - 1, j + 1], "UR", True)
        self.color_coins(value, [i + 1, j + 1], "DR", True)
        self.color_coins(value, [i - 1, j - 1], "UL", True)
        self.color_coins(value, [i + 1, j - 1], "DL", True)

        self.game_over = self.is_terminal()

        reward = [0.0] * self.num_players

        if self.game_outcome is not None:
            for player in range(self.num_players):
                if player == self.game_outcome:
                    reward[player] += 1.0
                else:
                    reward[player] -= 1.0

        self.update_current_player()

        return np.array(reward)

    def color_coins(self, curr_turn, curr_posn, direction, do_color):
        i = curr_posn[0]
        j = curr_posn[1]

        if i > 7 or i < 0 or j > 7 or j < 0:
            return False
        elif self.current_state[i][j] == 0:
            return False

        if self.current_state[i][j] == curr_turn:
            return True
        else:
            if direction == "U":
                new_posn = [i - 1, j]
            elif direction == "D":
                new_posn = [i + 1, j]
            elif direction == "R":
                new_posn = [i, j + 1]
            elif direction == "L":
                new_posn = [i, j - 1]
            elif direction == "UR":
                new_posn = [i - 1, j + 1]
            elif direction == "DR":
                new_posn = [i + 1, j + 1]
            elif direction == "UL":
                new_posn = [i - 1, j - 1]
            elif direction == "DL":
                new_posn = [i + 1, j - 1]

            ret = self.color_coins(curr_turn, new_posn, direction, do_color)
            if ret:
                if do_color:
                    self.current_state[i][j] = curr_turn

            return ret

    def get_actions(self, curr_player=-1):
        actions_list = []

        if curr_player == -1:
            value = self.current_player + 1
        else:
            value = curr_player

        curr_board = self.current_state
        for i in range(8):
            for j in range(8):
                if curr_board[i][j] == 0:
                    possible_count = 0

                    if i >= 1 and curr_board[i - 1][j] != value:
                        possible_count += int(self.color_coins(value, [i - 1, j], "U", False))
                    if i <= 6 and curr_board[i + 1][j] != value:
                        possible_count += int(self.color_coins(value, [i + 1, j], "D", False))
                    if j <= 6 and curr_board[i][j + 1] != value:
                        possible_count += int(self.color_coins(value, [i, j + 1], "R", False))
                    if j >= 1 and curr_board[i][j - 1] != value:
                        possible_count += int(self.color_coins(value, [i, j - 1], "L", False))
                    if i >= 1 and j <= 6 and curr_board[i - 1][j + 1] != value:
                        possible_count += int(self.color_coins(value, [i - 1, j + 1], "UR", False))
                    if i <= 6 and j <= 6 and curr_board[i + 1][j + 1] != value:
                        possible_count += int(self.color_coins(value, [i + 1, j + 1], "DR", False))
                    if i >= 1 and j >= 1 and curr_board[i - 1][j - 1] != value:
                        possible_count += int(self.color_coins(value, [i - 1, j - 1], "UL", False))
                    if i <= 6 and j >= 1 and curr_board[i + 1][j - 1] != value:
                        possible_count += int(self.color_coins(value, [i + 1, j - 1], "DL", False))

                    if possible_count > 0:
                        action = {'position': [i, j], 'value': self.current_player + 1}
                        actions_list.append(action)

        return actions_list

    def get_value_bounds(self):
        return {'defeat': -1, 'victory': 1,
                'min non-terminal': 0, 'max non-terminal': 0,
                'pre-computed min': -1, 'pre-computed max': 1,
                'evaluation function': None}

    def is_terminal(self):
        for_player_1 = self.get_actions(1)
        for_player_2 = self.get_actions(2)

        if len(for_player_1) > 1 or len(for_player_2) > 1:
            return False
        else:
            coin_count = [0] * self.num_players
            for i in range(8):
                for j in range(8):
                    coin_count[self.current_state[i][j] - 1] += 1

            if coin_count[0] == coin_count[1]:
                self.game_outcome = None
            else:
                if coin_count[0] > coin_count[1]:
                    self.game_outcome = 0
                else:
                    self.game_outcome = 1

            return True

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(str(self.current_state))

    def __str__(self):
        output = ''
        for elem in self.current_state:
            output += str(elem) + '\n'
        return output
