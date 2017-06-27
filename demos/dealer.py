import timeit
import multiprocessing

class DealerClass:
    def __init__(self, simulator, agents_list, num_simulations, sim_horizon, verbose=False):
        self.simulator = simulator
        self.player_list = agents_list

        self.player_count = len(agents_list)
        self.simulation_count = num_simulations

        self.simulation_history = []
        self.game_winner_list = []
        self.avg_move_time = []  # time taken per move by each player over all games

        self.verbose = verbose
        self.simulation_horizon = sim_horizon

    def start_simulation(self, multiprocess=True):
        if multiprocess:
            queues = []
            jobs = []

            for sim_num in range(self.simulation_count):
                q = multiprocessing.Queue()
                queues.append(q)  # our job's output will go here

                j = multiprocessing.Process(target=self.run_trial, args=(q, ))
                jobs.append(j)
                j.start()
            for j in jobs:  # wait for each job to finish
                j.join()
            for q in queues:
                [winner, game_history, time_sums] = q.get()
                self.game_winner_list.append(winner)
                self.write_simulation_history(game_history)
                self.avg_move_time += time_sums
        else:
            for sim_num in range(self.simulation_count):
                    [winner, game_history, time_sums] = self.run_trial()
                    self.game_winner_list.append(winner)
                    self.write_simulation_history(game_history)
                    self.avg_move_time += time_sums
        self.avg_move_time = [x / self.simulation_count for x in self.avg_move_time]  # average

    def run_trial(self, q=None):
        """Run a single simulation using the current configuration.

        :param q: an optional multiprocessing.Queue structure that allows for communication of results.
        """
        current_state = self.simulator.clone()
        game_history = []
        time_values = []

        h = 0
        while current_state.is_terminal() is False and h < self.simulation_horizon:
            actual_agent_id = current_state.get_current_player()

            # ASK FOR AN ACTION FROM THE AGENT. MOVE TIME CALCULATION
            move_start_time = timeit.default_timer()
            action_to_take = self.player_list[actual_agent_id].select_action(current_state)
            move_end_time = timeit.default_timer()

            if self.verbose:
                print(current_state)
                print("Agent {}".format(actual_agent_id + 1))
                print("Time for last move: {}".format(move_end_time - move_start_time))

            # ADD THE TIME VALUES TO A VARIABLE TO CALCULATE AVG TIME PER MOVE
            time_values.append([actual_agent_id, move_end_time - move_start_time])

            # TAKE THE RETURNED ACTION ON SIMULATOR
            reward = current_state.take_action(action_to_take)
            game_history.append([reward, action_to_take])

            h += 1

        if self.verbose:
            print(current_state)
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

        # time_sums = average time taken by each player per move
        for sum_value in range(len(time_sums)):
            time_sums[sum_value] = time_sums[sum_value] / moves_per_player

        if q is not None:  # if we are using multiprocessing
            q.put([winner, game_history, time_sums])
        return winner, game_history, time_sums

    def simulation_stats(self):
        return self.simulation_history, self.game_winner_list, self.avg_move_time

    def write_simulation_history(self, game_history):
        self.simulation_history.append(game_history)
