from agents import mcts_agent
from bandits import ucb_bandit_alg
from heuristics import rollout_heuristic


class UCTAgentClass(mcts_agent.MCTSAgentClass):
    my_name = "UCT Agent"

    def __init__(self, depth, max_width, num_trials, c=1, base_policy=None):
        h1 = rollout_heuristic.RolloutHeuristicClass(width=1, depth=depth, rollout_policy=base_policy)
        base_policy = h1.rollout_policy

        mcts_agent.MCTSAgentClass.__init__(self, depth=depth, max_width=max_width, num_trials=num_trials,
                                           heuristic=h1,
                                           bandit_alg_class=ucb_bandit_alg.UCBBanditAlgClass, bandit_parameters=c)

        self.agent_name = self.my_name + " (d={}, w={}, trials={}, c={}, base policy={})".format(depth,
                                                                                                 max_width,
                                                                                                 num_trials, c,
                                                                                                 base_policy.agent_name)
