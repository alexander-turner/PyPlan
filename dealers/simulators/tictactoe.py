import copy
import numpy as np
from abstract import abstract_state

"""
Simulator Class for TicTacToe

NOTE :
------

1. self.game_outcome = None -> Implies that the game is a draw. 
                                Otherwise this variable holds the winning player's number.
                                Player number 0 for X. 1 for O.

2. Reward scheme : Win = +3.0. Lose = -3.0. Draw = 0. Every move = -1
"""


class TicTacToeState(abstract_state.AbstractState):
    env_name = "Tic-Tac-Toe"
    num_players = 2
    current_player = 0
    original_state = [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]

    def __init__(self):
        self.current_state = self.original_state
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
        self.current_state = sim.current_state
        self.game_outcome = sim.game_outcome
        self.game_over = sim.game_over

    def take_action(self, action):
        position = action['position']
        self.current_state[position[0]][position[1]] = action['value']
        self.game_over = self.is_terminal()

        rewards = [0.0] * self.num_players

        if self.game_outcome is not None:
            rewards = [1.0 if player_idx == self.game_outcome else -1.0
                       for player_idx in range(self.num_players)]
        self.update_current_player()
        return np.array(rewards)

    def get_actions(self):
        actions = []

        for x in range(len(self.current_state)):
            for y in range(len(self.current_state[0])):
                if self.current_state[x][y] == 0:
                    actions.append({'position': [x, y], 'value': self.current_player + 1})

        return actions

    def get_value_bounds(self):
        return {'defeat': -1, 'victory': 1,
                'min non-terminal': 0, 'max non-terminal': 0,
                'pre-computed min': -1, 'pre-computed max': 1,
                'evaluation function': None}

    def is_terminal(self):
        xcount = 0
        ocount = 0

        current_state_val = self.current_state

        # Horizontal check for hit
        for x in range(len(current_state_val)):
            for y in range(len(current_state_val[0])):
                if current_state_val[x][y] == 1:
                    xcount += 1
                elif current_state_val[x][y] == 2:
                    ocount += 1

            if xcount == 3:
                self.game_outcome = 0
                break
            elif ocount == 3:
                self.game_outcome = 1
                break
            else:
                xcount = 0
                ocount = 0

        # Vertical check for hit
        if self.game_outcome is None:
            for y in range(len(current_state_val[0])):
                for x in range(len(current_state_val)):
                    if current_state_val[x][y] == 1:
                        xcount += 1
                    elif current_state_val[x][y] == 2:
                        ocount += 1

                if xcount == 3:
                    self.game_outcome = 0
                    break
                elif ocount == 3:
                    self.game_outcome = 1
                    break
                else:
                    xcount = 0
                    ocount = 0

        # Diagonal One Check for Hit
        x = 0
        y = 0
        xcount = 0
        ocount = 0

        if self.game_outcome is None:
            while x < len(current_state_val):
                if current_state_val[x][y] == 1:
                    xcount += 1
                elif current_state_val[x][y] == 2:
                    ocount += 1

                x += 1
                y += 1

            if xcount == 3:
                self.game_outcome = 0
            elif ocount == 3:
                self.game_outcome = 1
            else:
                xcount = 0
                ocount = 0

        # Diagonal Two Check for Hit
        x = 0
        y = len(current_state_val[0]) - 1
        xcount = 0
        ocount = 0

        if self.game_outcome is None:
            while x < len(current_state_val):
                if current_state_val[x][y] == 1:
                    xcount += 1
                elif current_state_val[x][y] == 2:
                    ocount += 1

                x += 1
                y -= 1

            if xcount == 3:
                self.game_outcome = 0
            elif ocount == 3:
                self.game_outcome = 1
            else:
                xcount = 0
                ocount = 0

        if self.game_outcome is None:
            # Check if the board is full
            x = 0
            y = 0
            game_over = True

            for x in range(len(current_state_val)):
                for y in range(len(current_state_val[0])):
                    if current_state_val[x][y] == 0:
                        game_over = False
                        break

            return game_over
        else:
            return True

    def __eq__(self, other):
        return self.current_state == other.current_state

    def __hash__(self):
        return hash(str(self.current_state))

    def __str__(self):
        output = ''
        for elem in self.current_state:
            output += str(elem) + '\n'
        return output
