from abstract import absstate
#from actions import yahtzeeaction
#from states import yahtzeestate
import random
from itertools import combinations

"""
Simulator Class for Yahtzee
"""


class YahtzeeStateClass(absstate.AbstractState):
    def __init__(self, num_players):
        #self.current_state = yahtzeestate.YahtzeeStateClass()
        self.num_players = num_players
        self.winning_player = None
        self.game_over = False
        self.myname = "Yahtzee"

    def clone(self):
        new_sim_obj = YahtzeeStateClass(self.num_players)
        new_sim_obj.set(self)
        return new_sim_obj

    def reset_simulator(self):
        self.winning_player = None
        #self.current_state = yahtzeestate.YahtzeeStateClass()
        self.game_over = False

    def get_simulator_state(self):
        return self.current_state

    def set(self, sim):
        self.current_state = sim.current_state
        self.winning_player = sim.winning_player
        self.game_over = sim.game_over

    def change_turn(self):
        current_roll = self.current_state.get_current_state()["state_val"]["current_roll"]

        if current_roll == 3:
            new_turn = self.current_state.get_current_state()["current_player"] + 1
            new_turn %= self.num_players
            self.current_state.get_current_state()["state_val"]["current_roll"] = 0
        else:
            self.current_state.get_current_state()["state_val"]["current_roll"] += 1
            new_turn = self.current_state.get_current_state()["current_player"]

        if new_turn == 0:
            self.current_state.get_current_state()["current_player"] = self.num_players
        else:
            self.current_state.get_current_state()["current_player"] = new_turn

    def print_board(self):
        score_sheet = self.current_state.get_current_state()["state_val"]["score_sheet"]
        totals = self.total_scores(score_sheet)
        return str("SCORE SHEET : \n" + str(score_sheet) + "\n" + "PLAYER TOTALS IN THIS PLAY : " + str(totals))

    # YAHTZEE SPECIFIC FUNCTION
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

    # YAHTZEE SPECIFIC FUNCTION
    def total_scores(self, current_scores):
        # INPUT CURRENT SCORE SHEET. RETURNS A VECTOR OF
        # CURRENT TOTAL SCORES FOR ALL PLAYERS.
        totals = [0.0] * self.num_players
        for category in current_scores:
            for player in range(self.num_players):
                if category[player] is not None:
                    totals[player] += category[player]

        return totals

    def take_action(self, action):
        action_value = action.get_action()
        type = action_value['type']
        value = action_value['value']
        current_roll = self.current_state.get_current_state()["state_val"]["current_roll"]
        self.game_over = self.is_terminal()
        reward_vector = [0.0] * self.num_players

        if type == "NOOP":
            dice_config = self.current_state.get_current_state()["state_val"]["dice_config"]
            category_points = self.get_category_points(dice_config, value)
            player_num = self.current_state.get_current_state()["current_player"]
            score_sheet = self.current_state.get_current_state()["state_val"]["score_sheet"]
            score_sheet[value][player_num] = category_points

            if score_sheet[value][player_num] is not None:
                reward_vector[player_num] = score_sheet[value][player_num]
                # reward_vector = self.total_scores(self.current_state.get_current_state()["state_val"]["score_sheet"])

            # SET CURRENT ROLL TO 3 TO END THE TURNS
            self.current_state.get_current_state()["state_val"]["current_roll"] = 3
        elif type == "ROLL":
            for dice in range(len(value)):
                new_roll = random.randrange(1, 7)
                self.current_state.get_current_state()["state_val"]["dice_config"][value[dice]] = new_roll

        return reward_vector

    def get_actions(self):
        actions_list = []
        current_player = self.current_state.get_current_state()["current_player"]

        if self.current_state.get_current_state()["state_val"]["current_roll"] == 0:
            action = {'type': "ROLL", 'value': (0, 1, 2, 3, 4)}
            #actions_list.append(yahtzeeaction.YahtzeeActionClass(action))
        else:
            # FIRST : SELECTING ONE OF THE POSSIBLE CATEGORIES AT THAT STATE.
            for category in range(0, 13):
                if self.current_state.get_current_state()["state_val"]["score_sheet"][category][
                            current_player] is None:
                    action = {}
                    action['type'] = "NOOP"
                    action['value'] = category
                    #actions_list.append(yahtzeeaction.YahtzeeActionClass(action))

            if self.current_state.get_current_state()["state_val"]["current_roll"] < 3:
                # SECOND : ROLLING ALL POSSIBLE COMBINATIONS
                for vals in range(1, 6):
                    # 5C1, 5C2, 5C3, 5C4, 5C5
                    # DICE NUMBER IS ZERO BASED INDEX
                    dice_values = [0, 1, 2, 3, 4]
                    for comb in combinations(dice_values, vals):
                        action = {'type': "ROLL", 'value': comb}
                        #actions_list.append(yahtzeeaction.YahtzeeActionClass(action))

        return actions_list

    def is_terminal(self):
        game_done = True
        totals = [0.0] * self.num_players

        for category in range(len(self.current_state.get_current_state()["state_val"]["score_sheet"])):
            for player in range(self.num_players):
                if self.current_state.get_current_state()["state_val"]["score_sheet"][category][player] is None:
                    game_done = False
                    break
                else:
                    totals[player] += self.current_state.get_current_state()["state_val"]["score_sheet"][category][
                        player]

        if game_done:
            max_val = totals[0]
            max_player = 0
            for vals in range(1, self.num_players):
                if totals[vals] > max_val:
                    max_val = totals[vals]
                    max_player = vals
            self.winning_player = max_player + 1

        return game_done

    def get_current_player(self):
        return self.current_state.get_current_state()["current_player"]

    def __eq__(self):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(self.current_state)
