from agents import *
from dealers import pacman_dealer

"""
It's important to consider depth and pull budget on a per-problem basis.

For example, one would think that having greater rollout depth would improve the heuristic. However, run and observe.

Discussion: 
Imagine that Pacman is being pursued down a narrow corridor and a ghost is on his tail. If he goes right, he dies. If 
he goes left, he lives, and the ghost (with 0.8 probability) moves left as well. The policy then employs random 
rollout - at each step, there is an equal probability of going right and going left. If at any point he goes right, 
he dies. However, while Pacman is simulating random behavior, the ghosts move in his direction with 0.8
probability. Therefore, as depth increases, it becomes multiplicatively more likely that Pacman dies in this rollout. 
This random rollout is simulated up to depth steps. 

This simulation is then averaged over each of the pulls. Since going left incurs -1 reward (assuming no pellet is there) 
and all further steps also cost -1, Pacman thinks that trying to escape would just decrease reward further, so he dies. 

When a depth of 0 (that is, a strictly one-step greedy agent) and a sufficient pull budget are provided, Pacman avoids 
ghosts just fine.
"""

if __name__ == '__main__':
    uniform_rollout_d0 = uniform_rollout_agent.UniformRolloutAgent(depth=0, num_pulls=100)
    uniform_rollout_d10 = uniform_rollout_agent.UniformRolloutAgent(depth=10, num_pulls=100)

    pacman = pacman_dealer.Dealer(layout_repr='testClassic')
    pacman.run(agents=[uniform_rollout_d0, uniform_rollout_d10], num_trials=10)
