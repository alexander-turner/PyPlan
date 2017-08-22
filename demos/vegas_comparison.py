from agents.bandits import *
from demos.vegas import *

"""
We can compare the cumulative and simple regret metrics for different bandit algorithms via "slot machines" (bandit
problems where each arm has a certain probability of giving a reward, else 0; see demos/vegas/vegas.py).
"""

if __name__ == '__main__':
    bandits = [uniform_bandit.UniformBandit, e_bandit.EBandit, ucb_bandit.UCBBandit]

    """
    Here, we define three "slot machines":
    1) Most arms give 0.05 reward all of the time, while the tenth rarely gives 1 reward.
    2) Twenty arms, each with a greater reward than the last; however, only .1 probability of receiving the reward!
    3) A ten-armed machine where one arm rarely returns a positive reward, while the rest very rarely return a 
        negative one.
    """
    slot_machines = [vegas.BanditProblem([[1, 0.1] if i == 9 else [0.05, 1] for i in range(10)]),
                     vegas.BanditProblem([[i/20, 0.1] for i in range(20)]),
                     vegas.BanditProblem([[1, 0.01] if i == 0 else [-1, 0.001] for i in range(10)])]

    generate_regret_curves.generate_regret_curves(bandits=bandits, pull_max=1e6, slot_machines=slot_machines)
