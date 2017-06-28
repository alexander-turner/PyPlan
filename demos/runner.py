import demos.simulate as simulate
from simulators import *
from heuristics import *
from agents import *

if __name__ == '__main__':  # for multiprocessing compatibility
    rand_agent = random_agent.RandomAgentClass()

    h1 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=rand_agent, width=1, depth=10)

    h10 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=rand_agent, width=10, depth=10)

    u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=1, num_pulls=100, policy=rand_agent)
    nested_u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=3, num_pulls=10, policy=u_ro)

    e_ro = e_rollout_agent.ERolloutAgentClass(depth=1, num_pulls=100, epsilon=0.5, policy=rand_agent)

    ucb_ro_d100_n100_c1 = ucb_rollout_agent.UCBRolloutAgentClass(depth=100, num_pulls=100, c=1.0, policy=rand_agent)

    ss_d3 = sparse_sampling_agent.SparseSamplingAgentClass(depth=3, pulls_per_node=5, heuristic=h1)
    ss_d5 = sparse_sampling_agent.SparseSamplingAgentClass(depth=5, pulls_per_node=5, heuristic=h1)

    uct = uct_agent.UCTAgentClass(depth=10, max_width=1, num_trials=100, c=1)

    eroot_uct1000 = eroot_uct_agent.ERootUCTAgentClass(depth=10, max_width=1, num_trials=1000, c=1)

    policy_set = [u_ro, ss_d3]
    switch_agent = policy_switch_agent.PolicySwitchAgentClass(num_pulls=10, policies=policy_set)

    e_switch_agent = e_policy_switch_agent.EPolicySwitchAgentClass(num_pulls=10, epsilon=0.5, policies=policy_set)

    openai = openai_sim.OpenAIStateClass('FrozenLake-v0', u_ro,
                                         wrapper_target='Frozen_Lake', api_key='sk_brIgt2t3TLGjd0IFrWW9rw')
    openai.run(num_trials=2, multiprocess=True, do_render=False)

    pacman = pacman_sim.PacmanStateClass(layout_repr='testClassic', agents=[u_ro, ss_d3, switch_agent, e_switch_agent],
                                         use_graphics=True)
    #pacman.run(2)

    sim = connect4_sim.Connect4StateClass()   # seems to be playing same game each time almost
    #simulate.run(simulator=sim, agents=[switch_agent, nested_u_ro], num_trials=10)
