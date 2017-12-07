import numpy as np
from abstract import abstract_evaluation
from agents import random_agent


class RolloutEvaluation(abstract_evaluation.AbstractEvaluation):
    """Facilitates rollout according to a user-defined policy, width, and depth."""
    name = "Rollout Evaluation"

    def __init__(self, width=1, depth=10, rollout_policy=None):
        super().__init__(width=width, depth=depth)
        # If no policy is provided, initialize a random agent
        self.rollout_policy = rollout_policy if rollout_policy else random_agent.RandomAgent()

    def evaluate(self, state):
        """Evaluate the state using width, depth, and the rollout policy."""
        total_rewards = np.array([0.0] * state.num_players)
        for _ in range(self.width):  # for each of width simulations
            total_rewards += self.run_rollout(state)

        return total_rewards / self.width

    def run_rollout(self, state):
        """Simulate a rollout, returning the rewards accumulated."""
        rewards = np.array([0.0] * state.num_players)
        sim_state = state.clone()  # create the simulated state so that the current state is left unchanged
        for _ in range(self.depth):
            if sim_state.is_terminal():  # act and track rewards as long as possible
                break
            action = self.rollout_policy.select_action(sim_state)
            rewards += sim_state.take_action(action)

        return rewards
