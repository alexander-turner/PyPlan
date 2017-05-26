from abstract import absheuristic

# Given a set of policies, choose the best policy's choice at each state
class SwitchingHeuristicClass(absheuristic.AbstractHeuristic):
    myname = "Switching Heuristic"

    def __init__(self, switch_policies, width=1, depth=10):
        self.heuristicname = self.myname
        self.switch_policies = switch_policies
        self.width = width
        self.depth = depth

    def get_heuristic_name(self):
        return self.agentname

    # assumes is first player
    def evaluate(self, state):
        sim_state = state.clone()
        total_reward = [0]*sim_state.number_of_players()

        for sim_num in range(self.width):
            h=0
            sim_state.set(state)
            while sim_state.is_terminal() is False and h <= self.depth:
                # Find the best action out of the available policies
                best_reward = [float("-inf")]
                for i in range(1, sim_state.number_of_players()-1):
                    best_reward[i] = float("inf")

                for policy in self.switch_policies:
                    action_to_take = policy.select_action(sim_state)
                    if action_to_take is None:
                        continue
                    reward = sim_state.take_action(action_to_take)
                    if reward[0] > best_reward[0]:
                        best_reward = reward
                        best_action = action_to_take
                reward = best_reward
                total_reward = [sum(r) for r in zip(total_reward, reward)]
                h += 1

        return [r / self.width for r in total_reward]
