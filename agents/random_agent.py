from abstract import absagent
import random


class RandomAgentClass(absagent.AbstractAgent):
    """An agent which randomly chooses from the available actions at each state."""
    myname = "RANDOM"

    def __init__(self, rollout_policy=None):
        self.agentname = self.myname
        self.rollout_policy = rollout_policy

    @staticmethod
    def create_copy():
        return RandomAgentClass()

    def get_agent_name(self):
        return self.agentname

    def select_action(self, state):
        """Randomly select a valid action."""
        valid_actions = state.get_actions()
        actions_count = len(valid_actions)

        if actions_count == 0:
            raise ValueError("Action count cannot be zero.")

        choice = random.randrange(actions_count)

        return valid_actions[choice]
