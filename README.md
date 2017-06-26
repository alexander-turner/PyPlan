## Synopsis

PyPlan is an easily-extensible library of Reinforcement Learning algorithms and environments, with OpenAI Gym integration for discrete action spaces.

## Code Example

Running the Uniform Rollout bandit algorithm on [Berkeley's Pacman simulator](http://ai.berkeley.edu/project_overview.html) is as simple as:

```
rand_agent = random_agent.RandomAgentClass()
u_ro = uniform_rollout_agent.UniformRolloutAgentClass(depth=1, num_pulls=10, policy=rand_agent)

pacman = pacman_sim.PacmanStateClass('originalClassic', u_ro)
pacman.run()
```

To implement your own agents or environments, consult the files in `abstract/` and the included examples. A tutorial series based on this suite is currently in production.

## Installation

After cloning the project, simply run `demos\runner.py`. 

Unix and Mac users must follow the [OpenAI installation instructions](https://github.com/openai/gym#installation). 

Although OpenAI does not officially support Windows, Windows users may use the included `WinPython` interpreter (located in the `WinPython-64bit-3.6.1.0Zero\python-3.6.1.amd64\` subdirectory) to access the Gym's full functionality. Furthermore, to record episodes for games with video output (e.g. Atari), Windows users must [install `ffmpeg`](http://www.wikihow.com/Install-FFmpeg-on-Windows) - make sure to add the `bin` subdirectory to your **system** path!
