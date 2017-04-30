from abstract import absagent
import random


class RandomAgentClass(absagent.AbstractAgent):
    myname = "RANDOM"

    def __init__(self, rollout_policy = None):
        self.agentname = self.myname
        self.rollout_policy = rollout_policy

    def create_copy(self):
        return RandomAgentClass()

    def get_agent_name(self):
        return self.agentname

    def select_action(self, state):
        valid_actions = state.get_actions()
        actions_count = len(valid_actions)

        if actions_count == 0:
            raise ValueError("Action count cannot be zero.")

        if actions_count == 1:
            choice = 0
        else:
            choice = random.randrange(actions_count)

        return valid_actions[choice]