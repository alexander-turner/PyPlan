from agents import mcts_agent
from agents import random_agent
from bandits import ucb_bandit_alg
from bandits import e_bandit_alg
from heuristics import rollout_heuristic


class ERootUCTAgentClass(mcts_agent.MCTSAgentClass):
    myname = "0.5-Greedy Root UCT Agent"

    def __init__(self, depth, max_width, num_trials, c=1, base_policy=None):
        if base_policy is None:
            base_policy = random_agent.RandomAgentClass()

        h1 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=base_policy, width=1, depth=depth)

        mcts_agent.MCTSAgentClass.__init__(self, depth=depth, max_width=max_width, num_trials=num_trials,
                                           heuristic=h1,
                                           BanditAlgClass=ucb_bandit_alg.UCBBanditAlgClass, bandit_parameters=c,
                                           RootBanditAlgClass=e_bandit_alg.EBanditAlgClass,
                                           root_bandit_parameters=0.5)

        self.agentname = self.myname
