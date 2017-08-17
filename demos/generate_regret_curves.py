import copy
import matplotlib.pyplot as plt


def generate_regret_curves(agents, pull_max, simulator, num_trials=50):
    """Generate and graph regret curves for agents x pull_values on simulator.

    :param agents: one of each agent for which regret curves will be generated.
    :param pull_max: the maximum number of pulls to use.
    :param simulator: assume simulator outputs cumulative and simple regrets.
    :param num_trials: for how many trials we will run.
    """
    # Generate permutations of agents using pull values
    pull_increments = range(int(pull_max/10), int(pull_max * 1.1), int(pull_max/10))
    new_agents = {}
    for agent in agents:
        new_agents[agent] = {}  # sort by agent blueprint (for later display)
        for num_pulls in pull_increments:
            new_agent = copy.deepcopy(agent)
            new_agent.num_pulls = num_pulls
            new_agents[agent][new_agent] = new_agent

    # For each agent, initialize average regret
    cumulative, simple = {}, {}
    for key in new_agents:  # for each agent category
        cumulative[key], simple[key] = [0] * len(pull_increments), [0] * len(pull_increments)

    # Generate data
    for key in new_agents:
        for agent_id, agent in enumerate(new_agents[key]):  # different pull numbers
            for trial in range(1, num_trials + 1):
                c, s = trial, trial*2
                #c, s = simulator(agent)
                cumulative[key][agent_id] = (cumulative[key][-1] * (trial - 1) + c) / trial
                simple[key][agent_id] = (simple[key][-1] * (trial - 1) + s) / trial

    # Construct and display graphs
    plt.figure()

    # Derive y-axis bounds
    both_dicts = (cumulative, simple)  # todo use built in params for cleaner code (subplots)
    minimum_regret = min(min(min(current_dict[key]) for key in current_dict) for current_dict in both_dicts)
    maximum_regret = max(max(max(current_dict[key]) for key in current_dict) for current_dict in both_dicts)
    ylims = (minimum_regret, maximum_regret)

    for regret_type in ('Cumulative', 'Simple'):
        ax = plt.subplot(121 if regret_type == 'Cumulative' else 122)
        ax.grid('on')

        ax.set_xlim([pull_increments[0], pull_increments[-1]])
        ax.set_ylabel(regret_type + ' Regret')

        regret_dict = cumulative if regret_type == 'Cumulative' else simple
        ax.set_ylim(ylims)
        for key in new_agents:
            ax.plot(pull_increments, regret_dict[key], label=key.base_name)

        ax.legend(loc='lower right')
        ax.set_title(regret_type + ' Regret ({} trials)'.format(num_trials))

    plt.text(0, -4.5, 'Number of Pulls', ha='center')  # shared x-axis label
    plt.show()
