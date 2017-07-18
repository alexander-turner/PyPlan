from abstract import absstate
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


class TicTacToeState(absstate.AbstractState):
    original_state = {
            "state_val": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            "current_player": 1
        }

    def __init__(self):
        self.current_state = self.original_state
        self.num_players = 2
        self.winning_player = None
        self.game_over = False
        self.my_name = "Tic-Tac-Toe"

    def number_of_players(self):
        return self.num_players

    def take_action(self, action):
        position = action['position']
        value = action['value']
        self.current_state["state_val"][position[0]][position[1]] = value
        self.game_over = self.is_terminal()

        reward = [0.0] * self.num_players

        if self.winning_player is not None:
            for player in range(self.num_players):
                if player == self.winning_player:
                    reward[player] += 1.0
                else:
                    reward[player] -= 1.0

        return reward

    def get_actions(self):
        actions_list = []

        for x in range(len(self.current_state["state_val"])):
            for y in range(len(self.current_state["state_val"][0])):
                if self.current_state["state_val"][x][y] == 0:
                    actions_list.append({'position': [x, y], 'value': self.current_state["current_player"]})

        return actions_list

    def clone(self):
        new_sim_obj = TicTacToeState()
        new_sim_obj.set(self)
        return new_sim_obj

    def reinitialize(self):
        self.current_state = copy.deepcopy(self.original_state)
        self.winning_player = None
        self.game_over = False

    def set(self, sim):
        self.current_state = sim.current_state
        self.winning_player = sim.winning_player
        self.game_over = sim.game_over

    def change_turn(self):
        new_turn = self.current_state["current_player"] + 1
        new_turn %= self.num_players

        if new_turn == 0:
            self.current_state["current_player"] = self.num_players
        else:
            self.current_state["current_player"] = new_turn

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
                self.winning_player = 0
                break
            elif ocount == 3:
                self.winning_player = 1
                break
            else:
                xcount = 0
                ocount = 0

        # Vertical check for hit
        if self.winning_player is None:
            for y in range(len(current_state_val[0])):
                for x in range(len(current_state_val)):
                    if current_state_val[x][y] == 1:
                        xcount += 1
                    elif current_state_val[x][y] == 2:
                        ocount += 1

                if xcount == 3:
                    self.winning_player = 0
                    break
                elif ocount == 3:
                    self.winning_player = 1
                    break
                else:
                    xcount = 0
                    ocount = 0

        # Diagonal One Check for Hit
        x = 0
        y = 0
        xcount = 0
        ocount = 0

        if self.winning_player is None:
            while x < len(current_state_val):
                if current_state_val[x][y] == 1:
                    xcount += 1
                elif current_state_val[x][y] == 2:
                    ocount += 1

                x += 1
                y += 1

            if xcount == 3:
                self.winning_player = 0
            elif ocount == 3:
                self.winning_player = 1
            else:
                xcount = 0
                ocount = 0

        # Diagonal Two Check for Hit
        x = 0
        y = len(current_state_val[0]) - 1
        xcount = 0
        ocount = 0

        if self.winning_player is None:
            while x < len(current_state_val):
                if current_state_val[x][y] == 1:
                    xcount += 1
                elif current_state_val[x][y] == 2:
                    ocount += 1

                x += 1
                y -= 1

            if xcount == 3:
                self.winning_player = 0
            elif ocount == 3:
                self.winning_player = 1
            else:
                xcount = 0
                ocount = 0

        if self.winning_player is None:
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

    def set_current_player(self, player_index):
        self.current_state["current_player"] = player_index

    def __eq__(self, other):
        return self.current_state == other.current_state

    def __hash__(self):
        return hash(self.current_state)

    def __str__(self):
        output = ''
        for elem in self.current_state["state_val"]:
            output += str(elem) + '\n'
        return output
