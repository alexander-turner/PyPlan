from abstract import abstract_state
import random
import copy
from itertools import combinations

"""
Simulator Class for Yahtzee
"""


class YahtzeeState(abstract_state.AbstractState):
    original_state = {
            "state_val": {
                    "current_roll": 0,
                    "dice_config": [1] * 5,
                    "score_sheet": [[None] * 2 for _ in range(13)]
            },
            "current_player": 0
        }

    def __init__(self, num_players=2):
        self.current_state = copy.deepcopy(self.original_state)
        self.num_players = num_players
        self.game_outcome = None
        self.game_over = False
        self.my_name = "Yahtzee"

    def reinitialize(self):
        self.current_state = copy.deepcopy(self.original_state)
        self.game_outcome = None
        self.game_over = False

    def clone(self):
        return copy.deepcopy(self)

    def set(self, sim):
        self.current_state = copy.deepcopy(sim.current_state)
        self.game_outcome = sim.game_outcome
        self.game_over = sim.game_over

    def take_action(self, action):
        type = action['type']
        value = action['value']
        current_roll = self.current_state["state_val"]["current_roll"]
        self.game_over = self.is_terminal()
        reward_vector = [0.0] * self.num_players

        if type == "NOOP":
            dice_config = self.current_state["state_val"]["dice_config"]
            category_points = self.get_category_points(dice_config, value)
            player_num = self.current_state["current_player"]
            score_sheet = self.current_state["state_val"]["score_sheet"]
            score_sheet[value][player_num] = category_points

            if score_sheet[value][player_num] is not None:
                reward_vector[player_num] = score_sheet[value][player_num]
                reward_vector = self.total_scores(self.current_state["state_val"]["score_sheet"])

            # SET CURRENT ROLL TO 3 TO END THE TURNS
            self.current_state["state_val"]["current_roll"] = 3
        elif type == "ROLL":
            for dice in range(len(value)):
                new_roll = random.randrange(1, 7)
                self.current_state["state_val"]["dice_config"][value[dice]] = new_roll
        self.change_turn()
        return reward_vector

    def change_turn(self):
        current_roll = self.current_state["state_val"]["current_roll"]

        if current_roll == 3:
            new_turn = (self.current_state["current_player"] + 1) % self.num_players
            self.current_state["state_val"]["current_roll"] = 0
        else:
            self.current_state["state_val"]["current_roll"] += 1
            new_turn = self.current_state["current_player"]

        self.current_state["current_player"] = new_turn

    @staticmethod
    def get_category_points(dice_faces, category_num):
        counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        for face in dice_faces:
            counts[face] += 1

        if category_num == 0:
            return float(counts[1] * 1)
        elif category_num == 1:
            return float(counts[2] * 2)
        elif category_num == 2:
            return float(counts[3] * 3)
        elif category_num == 3:
            return float(counts[4] * 4)
        elif category_num == 4:
            return float(counts[5] * 5)
        elif category_num == 5:
            return float(counts[6] * 6)
        elif category_num == 6:
            hit = False
            for vals in counts:
                if counts[vals] == 3:
                    hit = True
            if hit:
                sum = 0.0
                for vals in counts:
                    sum += float(counts[vals] * vals)
                return sum
        elif category_num == 7:
            hit = False
            for vals in counts:
                if counts[vals] == 4:
                    hit = True
            if hit:
                sum = 0.0
                for vals in counts:
                    sum += float(counts[vals] * vals)
                return sum
        elif category_num == 8:
            # FULL HOUSE
            hit = True
            for vals in counts:
                if counts[vals] > 0:
                    if counts[vals] is not 2 and counts[vals] is not 3:
                        hit = False
                        break
            if hit:
                return 25.0
        elif category_num == 9:
            # SMALL STRAIGHT
            hit = False
            pointer1 = 1
            count = 0

            while pointer1 <= 3:
                if counts[pointer1] == 1:
                    count = 1
                    for pointer2 in range(pointer1 + 1, pointer1 + 4):
                        if counts[pointer2] >= 1:
                            count += 1
                        else:
                            break
                    if count >= 4:
                        return 30.0
                pointer1 += 1
        elif category_num == 10:
            # LARGE STRAIGHT
            hit = False
            pointer1 = 1
            count = 0
            while pointer1 <= 2:
                if counts[pointer1] == 1:
                    count = 1
                    for pointer2 in range(pointer1 + 1, pointer1 + 5):
                        if counts[pointer2] == 1:
                            count += 1
                        else:
                            return 0.0

                    if count >= 5:
                        return 40.0

                pointer1 += 1
        elif category_num == 11:
            # Chance
            sum = 0
            for vals in counts:
                sum += counts[vals] * vals
            return float(sum)
        elif category_num == 12:
            # Yahtzee
            for vals in counts:
                if counts[vals] == 5:
                    return 50.0
        return 0.0

    # Yahtzee-specific function
    def total_scores(self, current_scores):
        """Given a score sheet, returns a list of total scores for all players."""
        totals = [0.0] * self.num_players
        for category in current_scores:
            for player in range(self.num_players):
                if category[player] is not None:
                    totals[player] += category[player]

        return totals

    def get_actions(self):
        actions_list = []
        current_player = self.current_state["current_player"]

        if self.current_state["state_val"]["current_roll"] == 0:
            actions_list.append({'type': "ROLL", 'value': (0, 1, 2, 3, 4)})
        else:
            # Select one of the possible categories at the state
            for category in range(0, 13):
                if self.current_state["state_val"]["score_sheet"][category][current_player] is None:
                    actions_list.append({'type': "NOOP", 'value': category})

            if self.current_state["state_val"]["current_roll"] < 3:
                # Roll all possible combinations
                for vals in range(1, 6):
                    # 5C1, 5C2, 5C3, 5C4, 5C5
                    # Dice number is 0-indexed
                    dice_values = [0, 1, 2, 3, 4]
                    for comb in combinations(dice_values, vals):
                        actions_list.append({'type': "ROLL", 'value': comb})

        return actions_list

    def number_of_players(self):
        return self.num_players

    def get_current_player(self):
        return self.current_state["current_player"]

    def set_current_player(self, player_index):
        self.current_state["current_player"] = player_index

    def get_value_bounds(self):
        return {'defeat': 0, 'victory': 0,
                'min non-terminal': 0, 'max non-terminal': 50,
                'pre-computed min': None, 'pre-computed max': None,
                'evaluation function': None}

    def is_terminal(self):
        game_done = True
        totals = [0.0] * self.num_players

        for category in range(len(self.current_state["state_val"]["score_sheet"])):
            for player in range(self.num_players):
                if self.current_state["state_val"]["score_sheet"][category][player] is None:
                    game_done = False
                    break
                else:
                    totals[player] += self.current_state["state_val"]["score_sheet"][category][player]

        if game_done:
            max_val = totals[0]
            max_player = 0
            for vals in range(1, self.num_players):
                if totals[vals] > max_val:
                    max_val = totals[vals]
                    max_player = vals
            self.game_outcome = max_player

        return game_done

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(str(self.current_state["state_val"]))

    def __str__(self):
        score_sheet = self.current_state["state_val"]["score_sheet"]
        totals = self.total_scores(score_sheet)
        return str("Score sheet: \n" + str(score_sheet) + "\n" + "Player totals in this play: " + str(totals))
