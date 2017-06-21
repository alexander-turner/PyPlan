## Synopsis

PyPlan is an easily-extensible library of Reinforcement Learning algorithms and environments, with OpenAI Gym integration for discrete environments.

## Code Example

Running the Uniform Rollout bandit algorithm on [Berkeley's Pacman simulator](http://ai.berkeley.edu/project_overview.html) is as simple as:

```
rand_agent = random_agent.RandomAgentClass()
u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=1, num_pulls=10, policy=rand_agent)

pacman = pacman_sim.PacmanStateClass('originalClassic', u_ro)
pacman.run()
```

## Installation

After cloning the project, simply run `runner.py`. Unix and Mac users must then follow the [OpenAI installation instructions](https://github.com/openai/gym#installation) for the installation at `simulations/gym-master`. As the Gym will be updated with new environments, the `gym-master` directory can safely be manually updated as the user sees fit.

### OpenAI for Windows

Although OpenAI does not officially support Windows, the [`algorithmic`](https://gym.openai.com/envs#algorithmic), [`classic_control`](https://gym.openai.com/envs#classic_control), [`parameter_tuning`](https://gym.openai.com/envs#parameter_tuning), and [`toy_text`](https://gym.openai.com/envs#toy_text) environment suites can still be used. 

1) [Install Bash for Windows](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/).
2) Download the [`numpy`](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy) and [`scipy`](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy) wheel files.
3) Fill in the filepaths in `OpenAI_Win_setup.sh`.
4) Run `OpenAI_Win_setup.sh`.

----

To implement your own agents or environments, consult the files in `abstract/` and the included examples. A tutorial series based on this suite is currently in production.
