from abstract import abstract_state
import copy

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
    original_state = {
            "state_val": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "current_player": 1
        }

    def __init__(self):
        self.current_state = self.original_state
        self.num_players = 2
        self.game_outcome = None
        self.game_over = False
        self.my_name = "Tic-Tac-Toe"

    def number_of_players(self):
        return self.num_players

    def take_action(self, action):
        position = action['position']
        self.current_state['state_val'][position[0]][position[1]] = action['value']
        self.game_over = self.is_terminal()

        reward = [0.0] * self.num_players

        if self.game_outcome is not None:
            for player in range(self.num_players):
                if player == self.game_outcome:
                    reward[player] += 1.0
                else:
                    reward[player] -= 1.0

        self.change_turn()

        return reward

    def get_actions(self):
        actions_list = []

        for x in range(len(self.current_state["state_val"])):
            for y in range(len(self.current_state["state_val"][0])):
                if self.current_state["state_val"][x][y] == 0:
                    actions_list.append({'position': [x, y], 'value': self.current_state["current_player"] + 1})

        return actions_list

    def clone(self):
        return copy.deepcopy(self)

    def reinitialize(self):
        self.current_state = copy.deepcopy(self.original_state)
        self.game_outcome = None
        self.game_over = False

    def set(self, sim):
        self.current_state = copy.deepcopy(sim.current_state)
        self.game_outcome = sim.game_outcome
        self.game_over = sim.game_over

    def change_turn(self):
        self.current_state["current_player"] = (self.current_state["current_player"] + 1) % self.num_players

    def is_terminal(self):
        xcount = 0
        ocount = 0

        current_state_val = self.current_state["state_val"]
        current_player = self.current_state["current_player"]

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

    def get_current_player(self):
        return self.current_state["current_player"]

    def get_value_bounds(self):
        return {'defeat': -1, 'min non-terminal': 0,
                'victory': 1, 'max non-terminal': 0,
                'pre-computed min': -1, 'pre-computed max': 1}

    def set_current_player(self, player_index):
        self.current_state["current_player"] = player_index

    def __eq__(self, other):
        return self.current_state == other.current_state

    def __hash__(self):
        return hash(str(self.current_state))

    def __str__(self):
        output = ''
        for elem in self.current_state["state_val"]:
            output += str(elem) + '\n'
        return output
