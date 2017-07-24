from abstract import abstract_heuristic
from agents.frameworks import random_agent
import multiprocessing
import numpy as np


class RolloutHeuristicClass(abstract_heuristic.AbstractHeuristic):
    """Facilitates rollout according to a user-defined policy, width, and depth."""
    my_name = "Rollout Heuristic"

    def __init__(self, width=1, depth=10, rollout_policy=None):
        """If no policy is provided, initialize a random agent."""
        self.heuristic_name = self.my_name
        self.width = width
        self.depth = depth
        if rollout_policy is None:
            self.rollout_policy = random_agent.RandomAgentClass()
        else:
            self.rollout_policy = rollout_policy

    def get_heuristic_name(self):
        return self.heuristic_name

    def evaluate(self, state):
        """Evaluate the state using the parameters of the heuristic and the rollout policy."""
        total_rewards = [0] * state.number_of_players()
        for sim_num in range(self.width):  # for each of width simulations
            total_rewards = [sum(r) for r in zip(total_rewards, self.run_rollout(state))]

        return [r / self.width for r in total_rewards]  # average rewards over each of width simulations

    def run_rollout(self, state):
        """Simulate a rollout, returning the rewards accumulated."""
        h = 0  # depth counter
        rewards = [0] * state.number_of_players()
        sim_state = state.clone()  # create the simulated state so that the current state is left unchanged
        while sim_state.is_terminal() is False and h <= self.depth:  # act and track rewards as long as possible
            action = self.rollout_policy.select_action(sim_state)
            immediate_rewards = sim_state.take_action(action)
            rewards = [sum(r) for r in zip(rewards, immediate_rewards)]
            h += 1

        return rewards
