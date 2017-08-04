## Synopsis

PyPlan is an easily-extensible library of planning algorithms and environments, with OpenAI Gym integration for environments with discrete action spaces.

## Code Example

Running a one-step greedy bandit algorithm on [Berkeley's Pacman simulator](http://ai.berkeley.edu/project_overview.html) is as simple as:

```
u_ro = uniform_rollout_agent.UniformRolloutAgent(depth=0, num_pulls=100)

pacman = pacman_dealer.Dealer(layout_representation='originalClassic')
pacman.run(agents=[u_ro], num_trials=10)
```

To implement your own agents or environments, consult the documentation in `abstract/`. A tutorial series based on this suite is currently in production.

## Installation

After cloning the project, get started with the files in the `demos/` directory. 

Unix and Mac users must follow the [OpenAI installation instructions](https://github.com/openai/gym#installation). If you get runtime errors, you may need to replace the installed `gym` folder with `WinPython/python-3.6.1.amd64/Lib/site-packages/gym`.

Although OpenAI does not officially support Windows, Windows users may use the included `WinPython` interpreter (located in the `WinPython/python-3.6.1.amd64/` subdirectory) to access the Gym's full functionality. Furthermore, to record episodes for games with video output, Windows users must [install `ffmpeg`](http://www.wikihow.com/Install-FFmpeg-on-Windows) - make sure to add the `bin` subdirectory to your **system** path!
