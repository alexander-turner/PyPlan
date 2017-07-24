from agents import *
from dealers import pacman_dealer, openai_dealer

"""
Although Pacman and Ms. Pacman are nearly identical games, the latter's simulator has a much finer grid. This multiplies
 the depth required to have the same level of lookahead.
"""

if __name__ == '__main__':
    u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=0, num_pulls=100)

    pacman = pacman_dealer.Dealer(layout_representation='originalClassic')
    pacman.run(agents=[u_ro], num_trials=1, multiprocess_mode='', show_moves=True)

    # be patient - Ms. Pacman has much more overhead!
    openai_pacman = openai_dealer.Dealer(env_name='MsPacman-v0')
    openai_pacman.run(agents=[u_ro], num_trials=1, multiprocess_mode='', show_moves=True)
