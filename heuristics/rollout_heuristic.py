from abstract import absheuristic

class RolloutHeuristicClass(absheuristic.AbstractHeuristic):
    myname = "Rollout Heuristic"

    def __init__(self, rollout_policy, width=1, depth=10):
        self.heuristicname = self.myname
        self.rollout_policy = rollout_policy
        self.width = width
        self.depth = depth

    def get_heuristic_name(self):
        return self.agentname

    def evaluate(self, state):
        sim_state = state.clone()
        total_reward = [0]*sim_state.number_of_players()

        for sim_num in range(self.width):
            h = 0
            sim_state = state.clone()
            while sim_state.is_terminal() is False and h <= self.depth:
                action_to_take = self.rollout_policy.select_action(sim_state)
                reward = sim_state.take_action(action_to_take)
                if reward:
                    total_reward = [sum(r) for r in zip(total_reward, reward)]
                    h += 1
                else:
                    break  # no longer a viable state; reset

        return [r / self.width for r in total_reward]