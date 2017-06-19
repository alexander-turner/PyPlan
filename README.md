## Synopsis

PyPlan is an easily-extensible library of Reinforcement Learning algorithms and environments, with OpenAI Gym integration planned for the near future.

## Code Example

Running the Uniform Rollout bandit algorithm on [Berkeley's Pacman simulator](http://ai.berkeley.edu/project_overview.html) is as simple as:

```
rand_agent = random_agent.RandomAgentClass()
u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=1, num_pulls=10, policy=rand_agent)

pacman = pacman_sim.PacmanStateClass('originalClassic', u_ro)
pacman.run()
```

## Installation

After cloning the project, simply run `runner.py`. To implement your own agents or environments, consult the files in `abstract/` and the included examples. A tutorial series based on this suite is currently in production.
