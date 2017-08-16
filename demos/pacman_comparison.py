from agents import *
from dealers import pacman_dealer
from demos import generate_regret_curves

"""

"""

if __name__ == '__main__':
    u_ro = uniform_rollout_agent.UniformRolloutAgent(depth=1, num_pulls=10)
    e_ro = e_rollout_agent.ERolloutAgent(depth=1, num_pulls=10)
    ucb_ro = ucb_rollout_agent.UCBRolloutAgent(depth=1, num_pulls=10, c=1.0)
    agent_list = [u_ro, e_ro, ucb_ro]
    pull_values = [10, 50, 100]

    pacman = pacman_dealer.Dealer(layout_repr='testClassic')
    generate_regret_curves.generate_regret_curves(agent_list, pull_values, pacman)
    """
    u_ro_10 = uniform_rollout_agent.UniformRolloutAgent(depth=1, num_pulls=10)
    u_ro_50 = uniform_rollout_agent.UniformRolloutAgent(depth=1, num_pulls=50)
    u_ro_100 = uniform_rollout_agent.UniformRolloutAgent(depth=1, num_pulls=100)

    e_ro_10 = e_rollout_agent.ERolloutAgent(depth=1, num_pulls=10)
    e_ro_50 = e_rollout_agent.ERolloutAgent(depth=1, num_pulls=50)
    e_ro_100 = e_rollout_agent.ERolloutAgent(depth=1, num_pulls=100)

    ucb_ro_10 = ucb_rollout_agent.UCBRolloutAgent(depth=1, num_pulls=10, c=1.0)
    ucb_ro_50 = ucb_rollout_agent.UCBRolloutAgent(depth=1, num_pulls=50, c=1.0)
    ucb_ro_100 = ucb_rollout_agent.UCBRolloutAgent(depth=1, num_pulls=100, c=1.0)

    agent_list = [u_ro_10, u_ro_50, u_ro_100,
                  e_ro_10, e_ro_50, e_ro_100,
                  ucb_ro_10, ucb_ro_50, ucb_ro_100]

    pacman = pacman_dealer.Dealer(layout_representation='testClassic')
    pacman.run(agents=agent_list, num_trials=20)
    """
