from agents import *
from dealers import *
from agents.evaluations import *

if __name__ == '__main__':  # for multiprocessing compatibility
    # Dealer objects
    openai = openai_dealer.Dealer()
    pacman = pacman_dealer.Dealer(layout_repr='testClassic')
    native = native_dealer.Dealer(simulation_horizon=50)

    random = random_agent.RandomAgent()
    u_ro = uniform_rollout_agent.UniformRolloutAgent(depth=0, num_pulls=10)
    nested_u_ro = uniform_rollout_agent.UniformRolloutAgent(depth=2, num_pulls=10, policy=u_ro)

    e_ro = e_rollout_agent.ERolloutAgent(depth=1, num_pulls=10)

    ucb_ro = ucb_rollout_agent.UCBRolloutAgent(depth=1, num_pulls=100, c=1.0)

    evaluation = rollout_evaluation.RolloutEvaluation(width=1, depth=10)
    ss_d2 = sparse_sampling_agent.SparseSamplingAgent(depth=2, pulls_per_node=20, evaluation=evaluation)
    ss_d5 = sparse_sampling_agent.SparseSamplingAgent(depth=5, pulls_per_node=5, evaluation=evaluation)

    uct = uct_agent.UCTAgent(depth=2, max_width=1, num_trials=10, c=1)
    e_root_uct = e_root_uct_agent.ERootUCTAgent(depth=10, max_width=1, num_trials=1000, c=1)
    fsss = fsss_agent.FSSSAgent(depth=3, pulls_per_node=20, num_trials=100)

    policy_set = [u_ro, e_ro]
    switch_agent = policy_switching_agent.PolicySwitchingAgent(depth=2, num_pulls=10, policies=policy_set)
    e_switch_agent = e_policy_switching_agent.EPolicySwitchingAgent(depth=10, num_pulls=10, policies=policy_set)

    all_agents = [random, u_ro, nested_u_ro, e_ro, ucb_ro, ss_d2, ss_d5, fsss, uct, e_root_uct, switch_agent,
                  e_switch_agent]

    openai.run(agents=[uct], num_trials=10, env_name='FrozenLake-v0', multiprocess_mode='', show_moves=True, upload=False)

    #pacman.run(agents=[nested_u_ro], num_trials=6, multiprocess_mode='trials')

    #native.run(agents=[u_ro, fsss], num_trials=1, env_name='Chess', multiprocess_mode='trial', show_moves=True)
