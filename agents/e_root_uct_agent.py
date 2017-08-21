from agents import mcts_framework
from agents.bandits import ucb_bandit, e_bandit
from agents.evaluations import rollout_evaluation


class ERootUCTAgent(mcts_framework.MCTSFramework):
    base_name = "0.5-Greedy Root UCT Agent"

    def __init__(self, depth, max_width, num_trials, c=1, base_policy=None):
        evaluation = rollout_evaluation.RolloutEvaluation(width=1, depth=depth, rollout_policy=base_policy)
        base_policy = evaluation.rollout_policy

        mcts_framework.MCTSFramework.__init__(self, depth=depth, max_width=max_width, num_trials=num_trials,
                                              evaluation=evaluation,
                                              bandit_class=ucb_bandit.UCBBandit, bandit_parameters=c,
                                              root_bandit_class=e_bandit.EBandit,
                                              root_bandit_parameters=0.5)

        self.name = self.base_name + " (d={}, w={}, trials={}, c={}, base policy={})".format(depth, max_width,
                                                                                             num_trials, c,
                                                                                             base_policy.name)

