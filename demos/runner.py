from agents import *
from heuristics import *
from dealers import *

if __name__ == '__main__':  # for multiprocessing compatibility
    # Dealer objects
    openai = openai_dealer.Dealer(api_key='sk_brIgt2t3TLGjd0IFrWW9rw')
    pacman = pacman_dealer.Dealer(layout_representation='testClassic')
    native = native_dealer.Dealer()

    h1 = rollout_heuristic.RolloutHeuristicClass(width=1, depth=10)
    h10 = rollout_heuristic.RolloutHeuristicClass(width=10, depth=10)

    u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=1, num_pulls=10)
    nested_u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=3, num_pulls=10, policy=u_ro)

    e_ro = e_rollout_agent.ERolloutAgentClass(depth=1, num_pulls=10, epsilon=0.5)

    ucb_ro = ucb_rollout_agent.UCBRolloutAgentClass(depth=10, num_pulls=1000, c=1.0)

    ss_d2 = sparse_sampling_agent.SparseSamplingAgentClass(depth=2, pulls_per_node=20, heuristic=h1)
    ss_d5 = sparse_sampling_agent.SparseSamplingAgentClass(depth=5, pulls_per_node=5, heuristic=h1)
    fsss = fsss_agent.FSSSAgentClass(depth=2, pulls_per_node=20)

    uct = uct_agent.UCTAgentClass(depth=10, max_width=1, num_trials=1000, c=1)
    e_root_uct = e_root_uct_agent.ERootUCTAgentClass(depth=10, max_width=1, num_trials=1000, c=1)

    policy_set = [u_ro, ss_d2]
    switch_agent = policy_switch_agent.PolicySwitchAgentClass(depth=10, num_pulls=10, policies=policy_set)
    e_switch_agent = e_policy_switch_agent.EPolicySwitchAgentClass(depth=10, num_pulls=10, policies=policy_set)

    all_agents = [u_ro, nested_u_ro, e_ro, ucb_ro, ss_d2, ss_d5, fsss, uct, e_root_uct, switch_agent, e_switch_agent]

    #openai.run(agents=[u_ro], num_trials=10, env_name='CartPole-v0', multiprocess=True, show_moves=False, upload=False)

    pacman.run(agents=[fsss, ss_d2], num_trials=2, multiprocess=False)

    #native.run(simulator_str='connect4', agents=[e_ro, u_ro], num_trials=5)


