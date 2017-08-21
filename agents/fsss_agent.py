from agents.frameworks import fsss_framework
from agents.evaluations import rollout_evaluation


class FSSSAgent(fsss_framework.FSSSFramework):
    """A Forward Search Sparse Sampling agent, as described by Walsh et al."""
    base_name = "Forward Search Sparse Sampling Agent"

    def __init__(self, depth, pulls_per_node, num_trials, evaluation=None):
        if evaluation is None:
            evaluation = rollout_evaluation.RolloutEvaluation(width=1, depth=10)  # default to random rollout

        fsss_framework.FSSSFramework.__init__(self, depth=depth, pulls_per_node=pulls_per_node,
                                              num_trials=num_trials, evaluation=evaluation)

        self.name = self.base_name + " (d={}, n={}, trials={}, base policy={})".format(depth, pulls_per_node,
                                                                                       num_trials,
                                                                                       evaluation.rollout_policy.name)
