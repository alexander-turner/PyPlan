from abstract import absagent
import math
import sys
import timeit
from bandits import uniform_bandit_alg
from bandits import e_bandit_alg
from heuristics import zero_heuristic


class RecursiveBanditAgentClass(absagent.AbstractAgent):
    myname = "Recursive Bandit"

    def __init__(self, depth, pulls_per_node, heuristic=None, BanditClass=None, bandit_parameters=None):
        self.agentname = self.myname
        self.num_nodes = 1
        self.depth = depth
        self.pulls_per_node = pulls_per_node
        if heuristic is not None:
            self.heuristic = zero_heuristic.ZeroHeuristicClass()
        else:
            self.heuristic = heuristic

        if BanditClass is not None:
            self.BanditClass = uniform_bandit.UniformBanditClass
        else:
            self.BanditClass = BanditClass

        self.bandit_parameters = bandit_parameters

    def get_agent_name(self):
        return self.agentname

    def select_action(self, state):
        self.num_nodes = 1
        (value,action) = self.estimateV(state, self.depth)
        return action

    def estimateV(self, state, depth):
        self.num_nodes += 1
        current_player = state.get_current_player()
        if depth == 0 or state.is_terminal():
            return self.heuristic.evaluate(state),None

        action_list = state.get_actions()
        num_actions = len(action_list)

        if self.bandit_parameters is None:
            bandit = self.BanditClass(num_actions)
        else:
            bandit = self.BanditClass(num_actions, self.bandit_parameters)

        current_state = state.clone()

        Qvalues = [[0]*state.number_of_players()]*num_actions

        for i in range(self.pulls_per_node):
            chosen_arm = bandit.select_pull_arm()

            current_state.set(state)
            immediate_reward = current_state.take_action(action_list[chosen_arm])
            future_reward = self.estimateV(current_state, depth-1)[0]
            total_reward = [sum(r) for r in zip(immediate_reward, future_reward)]
            Qvalues[chosen_arm] = [sum(r) for r in zip(Qvalues[chosen_arm], total_reward)]

            bandit.update(chosen_arm, total_reward[current_player-1])

        best_arm_index = bandit.select_best_arm()
        return [q / bandit.get_num_pulls(best_arm_index) for q in Qvalues[best_arm_index]], action_list[best_arm_index]