from agents import *
from simulators import *
from demos import simulate

"""
This demo highlights the performance differences between Uniform, e-Greedy, and Nested rollout agents. 
"""

if __name__ == '__main__':
    rand_agent = random_agent.RandomAgentClass()

    u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=10, num_pulls=20, policy=rand_agent)
    nested_u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=2, num_pulls=10, policy=u_ro)

    connect4 = connect4_sim.Connect4StateClass()
    simulate.run(simulator=connect4, agents_list=[u_ro, nested_u_ro], num_trials=10, show_moves=False)

    e_ro = e_rollout_agent.ERolloutAgentClass(depth=10, num_pulls=20, epsilon=0.5, policy=rand_agent)
    nested_e_ro = e_rollout_agent.ERolloutAgentClass(depth=2, num_pulls=10, epsilon=0.5, policy=e_ro)

    simulate.run(simulator=connect4, agents_list=[u_ro, e_ro], num_trials=10, show_moves=False)

    simulate.run(simulator=connect4, agents_list=[nested_u_ro, nested_e_ro], num_trials=10, show_moves=False)
