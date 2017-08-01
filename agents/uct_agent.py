from agents import mcts_framework
from bandits import ucb_bandit
from evaluations import rollout_evaluation


class UCTAgent(mcts_framework.MCTSFramework):
    name = "UCT Agent"

    def __init__(self, depth, max_width, num_trials, c=1, base_policy=None):
        evaluation = rollout_evaluation.RolloutEvaluation(width=1, depth=depth, rollout_policy=base_policy)
        base_policy = evaluation.rollout_policy

        mcts_framework.MCTSFramework.__init__(self, depth=depth, max_width=max_width, num_trials=num_trials,
                                              evaluation=evaluation,
                                              bandit_alg_class=ucb_bandit.UCBBandit, bandit_parameters=c)

        self.name += " (d={}, w={}, trials={}, c={}, base policy={})".format(depth,
                                                                             max_width,
                                                                             num_trials,
                                                                             c,
                                                                             base_policy.name)
