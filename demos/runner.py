from agents import *
from heuristics import *
from gym import envs
from simulators import *


def run_all_openai(agents, num_trials):
    """A helper function for running OpenAI on all available simulators."""
    all_environments = envs.registry.all()
    openai = openai_sim.OpenAIStateClass(sim_name='FrozenLake-v0')
    for env_idx, env in enumerate(all_environments):
        if should_skip(env.id):  # duplicate games / games that hang
            continue
        try:
            openai.change_sim(env.id)
        except:  # If the action space is continuous
            print("Continuous action space - skipping {}.".format(env.id))
            continue
        openai.run(agents=agents, num_trials=num_trials, multiprocess=False, show_moves=False)


def should_skip(game_id):
    """Returns True if the game should be skipped."""
    skip_games = ["Assault", "BankHeist", "CliffWalking"]  # games take a very long time
    skip_keywords = ["Deterministic", "Frameskip", "ram"]
    for key in skip_games + skip_keywords:
        if str.find(game_id, key) != -1:
            return True


if __name__ == '__main__':  # for multiprocessing compatibility
    rand_agent = random_agent.RandomAgentClass()

    h1 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=rand_agent, width=1, depth=10)

    h10 = rollout_heuristic.RolloutHeuristicClass(rollout_policy=rand_agent, width=10, depth=10)

    u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=1, num_pulls=1, policy=rand_agent)
    nested_u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=2, num_pulls=15, policy=u_ro)

    e_ro = e_rollout_agent.ERolloutAgentClass(depth=1, num_pulls=100, epsilon=0.5, policy=rand_agent)

    ucb_ro_d100_n100_c1 = ucb_rollout_agent.UCBRolloutAgentClass(depth=100, num_pulls=100, c=1.0, policy=rand_agent)

    ss_d2 = sparse_sampling_agent.SparseSamplingAgentClass(depth=2, pulls_per_node=20, heuristic=h1)
    ss_d5 = sparse_sampling_agent.SparseSamplingAgentClass(depth=5, pulls_per_node=5, heuristic=h1)

    uct = uct_agent.UCTAgentClass(depth=10, max_width=1, num_trials=100, c=1)
    e_root_uct1000 = eroot_uct_agent.ERootUCTAgentClass(depth=10, max_width=1, num_trials=1000, c=1)

    policy_set = [u_ro, ss_d2]
    switch_agent = policy_switch_agent.PolicySwitchAgentClass(num_pulls=10, policies=policy_set)
    e_switch_agent = e_policy_switch_agent.EPolicySwitchAgentClass(num_pulls=10, epsilon=0.5, policies=policy_set)

    run_all_openai(agents=[u_ro], num_trials=1)
    #openai = openai_sim.OpenAIStateClass(sim_name='FrozenLake-v0', api_key='sk_brIgt2t3TLGjd0IFrWW9rw')


    pacman = pacman_sim.PacmanStateClass(layout_repr='testClassic', use_graphics=True)
    #pacman.run(agents=[switch_agent, e_switch_agent], num_trials=10)

    sim = connect4_sim.Connect4StateClass()
    dealer = dealer.DealerClass()
    #dealer.run(simulator=sim, agents=[u_ro, u_ro], num_trials=20, multiprocess=True)  # TODO: Fix first agent's advantage


