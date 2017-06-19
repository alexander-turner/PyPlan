from abstract import absheuristic


class RolloutHeuristicClass(absheuristic.AbstractHeuristic):
    """Facilitates rollout according to a user-defined policy, width, and depth."""
    myname = "Rollout Heuristic"

    def __init__(self, rollout_policy, width=1, depth=10):
        self.heuristicname = self.myname
        self.rollout_policy = rollout_policy
        self.width = width
        self.depth = depth

    def get_heuristic_name(self):
        return self.agentname

    def evaluate(self, state):
        """Evaluate the state using the parameters of the heuristic and the rollout policy."""
        sim_state = state.clone()  # create the simulated state so that the current state is left unchanged
        total_reward = [0]*sim_state.number_of_players()

        for sim_num in range(self.width):  # for each of width simulations
            h = 0  # reset depth counter
            sim_state.set(state)  # reset state
            while sim_state.is_terminal() is False and h <= self.depth:  # act and track rewards as long as possible
                action_to_take = self.rollout_policy.select_action(sim_state)
                reward = sim_state.take_action(action_to_take)
                total_reward = [sum(r) for r in zip(total_reward, reward)]
                h += 1

        return [r / self.width for r in total_reward]  # average rewards over each of width simulations
