import timeit


class DealerClass:
    def __init__(self, agents_list, initial_state, num_simulations, sim_horizon,  verbose=False):
        self.initial_state = initial_state
        self.player_list = agents_list

        self.player_count = len(agents_list)
        self.simulation_count = num_simulations

        self.simulation_history = []
        self.game_winner_list = []

        self.verbose = verbose
        self.simulation_horizon = sim_horizon

    def start_simulation(self):
        current_state = self.initial_state.clone()
        total_time_sums = []

        for count in range(self.simulation_count):
            if self.verbose:
                print("Simulation {}:".format(count))

            game_history = []
            h = 0
            time_values = []

            if self.verbose:
                print(current_state)

            while current_state.is_terminal() is False and h < self.simulation_horizon:
                actual_agent_id = current_state.get_current_player()

                # ASK FOR AN ACTION FROM THE AGENT. MOVE TIME CALCULATION
                move_start_time = timeit.default_timer()
                action_to_take = self.player_list[actual_agent_id].select_action(current_state)
                move_end_time = timeit.default_timer()

                if self.verbose:
                    print("Agent {}".format(actual_agent_id + 1))
                    print("Time for last move: {}".format(move_end_time - move_start_time))

                # ADD THE TIME VALUES TO A VARIABLE TO CALCULATE AVG TIME PER MOVE
                time_values.append([actual_agent_id, move_end_time - move_start_time])

                # TAKE THE RETURNED ACTION ON SIMULATOR
                reward = current_state.take_action(action_to_take)
                game_history.append([reward, action_to_take])

                if self.verbose:
                    print(current_state)
                h += 1

            # ----------------------
            # STATISTICS FOR THE GAME
            # ----------------------
            total_reward = [0.0] * self.player_count
            for turn in range(len(game_history)):
                total_reward = [x + y for x, y in zip(total_reward, game_history[turn][0])]

            winner = current_state.game_outcome

            if self.verbose:
                print("\nRewards :", total_reward)
                print("Winner :", str(winner))
                print("-" * 50)

            # CALCULATE AVG TIME PER MOVE
            time_sums = [0.0] * self.player_count
            moves_per_player = float(h / self.player_count)
            for val in time_values:
                time_sums[val[0]] += val[1]

            # time_sums WILL HAVE AVERAGE OF TIME TAKEN BY EACH PLAYER PER MOVE (FOR GIVEN SIMULATION COUNT).
            for sum_value in range(len(time_sums)):
                time_sums[sum_value] = time_sums[sum_value] / moves_per_player

            total_time_sums += time_sums
            # --------------
            # END OF STATISTICS
            # ---------------
            self.game_winner_list.append(winner)
            self.write_simulation_history(game_history)
            current_state.initialize()

        return [x / self.simulation_count for x in total_time_sums]

    def simulation_stats(self):
        return self.simulation_history, self.game_winner_list

    def write_simulation_history(self, game_history):
        self.simulation_history.append(game_history)
