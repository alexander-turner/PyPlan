from agents import *
from dealers import pacman_dealer
from dealers.simulators import pacmancode

"""
Each pull made by Uniform Rollout simulates a base policy to a certain depth. This allows us to build upon the 
performance of existing policies without spending our own time on optimization.
"""
if __name__ == '__main__':
    greedy_agent = pacmancode.pacmanAgents.GreedyAgent()  # simulates each legal action once and chooses the best
    uniform_rollout_agent = uniform_rollout_agent.UniformRolloutAgent(depth=10, num_pulls=10, policy=greedy_agent)

    pacman = pacman_dealer.Dealer(layout_repr='testClassic')
    pacman.run(agents=[greedy_agent, uniform_rollout_agent], num_trials=15)
