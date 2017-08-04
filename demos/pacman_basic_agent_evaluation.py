from agents import *
from dealers import pacman_dealer
from dealers.simulators import pacmancode

"""
Dealers run agents on simulators and compare their performance. Below, we run three agents for five
trials each. By running agents on a state many times, we can approximate the value of their policy at that state. 
In this case, we're comparing how well the agents do at the start state of a small Pacman box which contains one ghost.
"""
if __name__ == '__main__':
    random = random_agent.RandomAgent()  # randomly selects a legal action
    left_turn = pacmancode.pacmanAgents.LeftTurnAgent()  # turns left whenever possible
    greedy = pacmancode.pacmanAgents.GreedyAgent()  # simulates each legal action once and chooses the best

    pacman = pacman_dealer.Dealer(layout_representation='testClassic')
    pacman.run(agents=[random, left_turn, greedy], num_trials=5, multiprocess_mode='', show_moves=True)
