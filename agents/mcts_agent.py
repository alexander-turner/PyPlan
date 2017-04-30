from abstract import absagent
import random
from bandits import uniform_bandit_alg
from heuristics import zero_heuristic

def multinomial(p):
    r = random.random()
    val = 0
    for i in range(len(p)):
        val += p[i]
        if r <= val:
            return i

class BanditNode():
    def __init__(self, state, transition_reward, action_list, bandit):
        self.state = state
        self.bandit = bandit
        self.num_nodes = 0
        self.action_list = action_list

        # each action is associated with a dictionary storing successor bandits/states
        # the key for each bandit is the state and the value is a list [n,c] where n is a BanditNode and
        # c is the number of times that node has been sampled

        self.children = [{} for i in range(len(action_list))]
        self.transition_reward = transition_reward

class MCTSAgentClass(absagent.AbstractAgent):
    myname = "MCTS"

    def __init__(self, depth, max_width, num_trials, heuristic = None, BanditAlgClass = None, bandit_parameters = None,
                 RootBanditAlgClass = None, root_bandit_parameters = None):
        self.agentname = self.myname
        self.depth = depth
        self.max_width = max_width
        self.num_trials = num_trials
        if heuristic == None:
            self.heuristic = zero_heuristic.ZeroHeuristicClass()
        else:
            self.heuristic = heuristic

        self.bandit_parameters = bandit_parameters

        if BanditAlgClass == None:
            self.BanditAlgClass = uniform_bandi_alg.UniformBanditAlgClass
        else:
            self.BanditAlgClass = BanditAlgClass

        if RootBanditAlgClass == None:
            self.RootBanditAlgClass = self.BanditAlgClass
        else:
            self.RootBanditAlgClass = RootBanditAlgClass

        self.root_bandit_parameters = root_bandit_parameters

    def get_agent_name(self):
        return self.agentname

    def select_action(self, state):
        if self.depth == 0 or state.is_terminal():
            return None

        self.num_nodes = 1

        action_list = state.get_actions()

        if self.root_bandit_parameters == None:
            bandit = self.RootBanditAlgClass(len(action_list))
        else:
            bandit = self.RootBanditAlgClass(len(action_list),self.root_bandit_parameters)

        root_node = BanditNode(state, 0, action_list, bandit)

        for i in range(self.num_trials):
            self.run_trial(root_node, self.depth)

#        if hasattr(root_node.bandit, 'ave_reward'):
#            print("Action Values: ", root_node.bandit.ave_reward)
#        print("Num Pulls: ", root_node.bandit.num_pulls)

        return root_node.action_list[root_node.bandit.select_best_arm()]

    def run_trial(self, node, depth):
        # We guarantee that the node pointed to by node will not be modified

        # leaf nodes are those with node.bandit == None

        if node.bandit == None:
            return self.heuristic.evaluate(node.state)

        current_player = node.state.get_current_player()

        action_index = node.bandit.select_pull_arm()

        # if we reach max children nodes then select random among children, otherwise generate a new one

        if len(node.children[action_index]) >= self.max_width:
            keys = list(node.children[action_index].keys())
            counts = [node.children[action_index][k][1] for k in keys]
            normalizer = sum(counts)
            counts = [c/normalizer for c in counts]
            successor_index = multinomial(counts)
            successor_node = node.children[action_index][keys[successor_index]][0]
            immediate_reward = successor_node.transition_reward
            total_reward = [x+y for (x,y) in zip(immediate_reward, self.run_trial(successor_node, depth-1))]
        else:
            successor_state = node.state.clone()
            immediate_reward = successor_state.take_action(node.action_list[action_index])
            successor_actions = successor_state.get_actions()

            if successor_state in node.children[action_index]:
                successor_node = node.children[action_index][successor_state][0]
                node.children[action_index][successor_state][1] += 1
                total_reward = [x+y for (x,y) in zip(immediate_reward, self.run_trial(successor_node, depth-1))]
            else:
                if successor_state.is_terminal() or depth == 1:
                    successor_bandit = None
                elif self.bandit_parameters == None:
                    successor_bandit = self.BanditAlgClass(len(successor_actions))
                else:
                    successor_bandit = self.BanditAlgClass(len(successor_actions),self.bandit_parameters)

                successor_node = BanditNode(successor_state, immediate_reward, successor_actions, successor_bandit)
                node.children[action_index][successor_state] = [successor_node,1]
                self.num_nodes += 1

                total_reward = [x + y for (x,y) in zip(immediate_reward, self.heuristic.evaluate(successor_state))]

        node.bandit.update(action_index, total_reward[current_player-1])
        return total_reward