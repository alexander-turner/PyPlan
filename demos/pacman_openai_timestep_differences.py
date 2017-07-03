from agents import *
from heuristics import rollout_heuristic
from simulators import openai_sim, pacman_sim

"""
Although Pacman and Ms. Pacman are nearly identical games, the latter's simulator has a much finer grid. This multiplies
 the depth required to be able to have the same level of lookahead.
"""

if __name__ == '__main__':
    rand_agent = random_agent.RandomAgentClass()
    u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=0, num_pulls=100, policy=rand_agent)

    pacman = pacman_sim.PacmanStateClass(layout_repr='originalClassic', use_graphics=True)
    pacman.run(agents=[u_ro], num_trials=1, show_moves=True, multiprocess=True)

    # be patient - Ms. Pacman has much more overhead!
    openai_pacman = openai_sim.OpenAIStateClass(sim_name='MsPacman-v0')
    openai_pacman.run(agents=[u_ro], num_trials=1, show_moves=True, multiprocess=False)
