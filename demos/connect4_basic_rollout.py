from agents import *
from dealers import native_dealer

"""
This demo highlights the performance differences between uniform, e-greedy, and nested rollout agents. 

Play with the parameters and see how performance changes!
"""

if __name__ == '__main__':
    u_ro = uniform_rollout_agent.UniformRolloutAgent(depth=10, num_pulls=20)
    nested_u_ro = uniform_rollout_agent.UniformRolloutAgent(depth=2, num_pulls=10, policy=u_ro)

    e_ro = e_rollout_agent.ERolloutAgent(depth=10, num_pulls=20, epsilon=0.5)
    nested_e_ro = e_rollout_agent.ERolloutAgent(depth=2, num_pulls=10, epsilon=0.5, policy=e_ro)

    dealer = native_dealer.Dealer()
    dealer.run(agents=[u_ro, nested_u_ro], num_trials=10, env_name='Connect4')
    dealer.run(agents=[u_ro, e_ro], num_trials=10, env_name='Connect4')
    dealer.run(agents=[nested_u_ro, nested_e_ro], num_trials=10, env_name='Connect4')
