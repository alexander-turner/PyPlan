from agents import recursive_bandit_framework
from agents.evaluations import rollout_evaluation


class RolloutAgent(recursive_bandit_framework.RecursiveBanditFramework):
    name = "Rollout Agent"

    def __init__(self, depth, num_pulls, policy=None, bandit_class=None, bandit_parameters=None):
        evaluation = rollout_evaluation.RolloutEvaluation(width=1, depth=depth, rollout_policy=policy)

        recursive_bandit_framework.RecursiveBanditFramework.__init__(self, depth=1, pulls_per_node=num_pulls,
                                                                     evaluation=evaluation, bandit_class=bandit_class,
                                                                     bandit_parameters=bandit_parameters)
