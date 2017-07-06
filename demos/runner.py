from agents import *
from heuristics import *
from simulators import *


if __name__ == '__main__':  # for multiprocessing compatibility
    rand_agent = random_agent.RandomAgentClass()

    h1 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=rand_agent, width=1, depth=10)
    h10 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=rand_agent, width=10, depth=10)

    u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=1, num_pulls=10, policy=rand_agent)
    nested_u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=3, num_pulls=10, policy=u_ro)

    e_ro = e_rollout_agent.ERolloutAgentClass(depth=1, num_pulls=10, epsilon=0.5, policy=rand_agent)

    ucb_ro = ucb_rollout_agent.UCBRolloutAgentClass(depth=10, num_pulls=1000, c=1.0, policy=rand_agent)

    ss_d2 = sparse_sampling_agent.SparseSamplingAgentClass(depth=2, pulls_per_node=20, heuristic=h1)
    ss_d5 = sparse_sampling_agent.SparseSamplingAgentClass(depth=5, pulls_per_node=5, heuristic=h1)

    uct = uct_agent.UCTAgentClass(depth=10, max_width=1, num_trials=1000, c=1)
    e_root_uct = eroot_uct_agent.ERootUCTAgentClass(depth=10, max_width=1, num_trials=1000, c=1)

    policy_set = [u_ro, ss_d2]
    switch_agent = policy_switch_agent.PolicySwitchAgentClass(num_pulls=10, policies=policy_set)
    e_switch_agent = e_policy_switch_agent.EPolicySwitchAgentClass(num_pulls=10, epsilon=0.5, policies=policy_set)

    all_agents = [u_ro, nested_u_ro, e_ro, ucb_ro, ss_d2, ss_d5, uct, e_root_uct, switch_agent, e_switch_agent]

    openai = openai_sim.Dealer(api_key='sk_brIgt2t3TLGjd0IFrWW9rw')
    openai.run(agents=[uct, u_ro], num_trials=10, env_name='FrozenLake-v0', multiprocess=True, show_moves=False, upload=True)

    pacman = pacman_sim.Dealer(layout_representation='testClassic')
    #pacman.run(agents=[uct], num_trials=10)

    sim = connect4_sim.Connect4State()
    dealer = dealer.Dealer()
    #dealer.run(simulator=sim, agents=[uct, e_ro], num_trials=20, multiprocess=True)


