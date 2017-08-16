from abstract import abstract_agent
import random


class RandomAgent(abstract_agent.AbstractAgent):
    """An agent which randomly chooses from the available actions at each state."""
    base_name = name = "Random Agent"

    @staticmethod
    def create_copy():
        return RandomAgent()

    def select_action(self, state):
        """Randomly select a valid action."""
        valid_actions = state.get_actions()
        actions_count = len(valid_actions)

        if actions_count == 0:
            raise ValueError("Action count cannot be zero.")

        choice = random.randrange(actions_count)

        return valid_actions[choice]
