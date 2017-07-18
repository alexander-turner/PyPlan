from abstract import absheuristic
import multiprocessing
import numpy as np


class RolloutHeuristicClass(absheuristic.AbstractHeuristic):
    """Facilitates rollout according to a user-defined policy, width, and depth."""
    my_name = "Rollout Heuristic"

    def __init__(self, rollout_policy, width=1, depth=10, multiprocess=True):
        self.heuristic_name = self.my_name
        self.rollout_policy = rollout_policy
        self.width = width
        self.depth = depth
        self.multiprocess = False

    def get_heuristic_name(self):
        return self.heuristic_name

    def evaluate(self, state):
        """Evaluate the state using the parameters of the heuristic and the rollout policy."""
        if self.multiprocess:
            # ensures that the system still runs smoothly
            pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1))
            total_rewards = pool.map(self.run_rollout, [state] * self.width)
            total_rewards = np.sum(total_rewards, axis=0)
        else:
            sim_state = state.clone()  # create the simulated state so that the current state is left unchanged
            total_rewards = [0] * sim_state.number_of_players()
            for sim_num in range(self.width):  # for each of width simulations
                h = 0  # reset depth counter
                sim_state.set(state)  # reset state
                while sim_state.is_terminal() is False and h <= self.depth:  # act and track rewards as long as possible
                    action = self.rollout_policy.select_action(sim_state)
                    rewards = sim_state.take_action(action)
                    total_rewards = [sum(r) for r in zip(total_rewards, rewards)]
                    h += 1

        return [r / self.width for r in total_rewards]  # average rewards over each of width simulations

    def run_rollout(self, state):
        """Simulate a rollout, returning the rewards accumulated."""
        h = 0  # depth counter
        rewards = [0] * state.number_of_players()
        sim_state = state.clone()  # simulate state
        while sim_state.is_terminal() is False and h <= self.depth:  # act and track rewards as long as possible
            action = self.rollout_policy.select_action(sim_state)
            immediate_rewards = sim_state.take_action(action)
            rewards = [sum(r) for r in zip(rewards, immediate_rewards)]
            h += 1

        return rewards
