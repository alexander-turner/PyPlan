import copy
import matplotlib.pyplot as plt


def generate_regret_curves(agents, pull_values, simulator, num_trials=50):
    """Generate and graph regret curves for agents x pull_values on simulator.

    :param agents: one of each agent for which regret curves will be generated.
    :param pull_values: a list of pull values (with which agents will be initialized).
    :param simulator: assume simulator outputs cumulative and simple regrets.
    :param num_trials: for how many trials we will run
    """
    # Generate permutations of agents using pull values
    new_agents = []
    for agent in agents:
        for num_pulls in pull_values:
            new_agent = copy.deepcopy(agent)
            new_agent.num_pulls = num_pulls
            new_agent.name =
            new_agents.append(new_agent)

    # For each agent, initialize average regret
    cumulative, simple = {}, {}
    for agent in new_agents:
        cumulative[agent], simple[agent] = [0], [0]

    for agent_id, agent in enumerate(new_agents):
        for trial in range(1, num_trials + 1):
            c, s = trial + agent_id, trial*2 + agent_id
            #c, s = simulator(agent)
            cumulative[agent].append((cumulative[agent][trial-1] * (trial - 1) + c) / trial)
            simple[agent].append((simple[agent][trial-1] * (trial - 1) + s) / trial)

    plt.figure(1)
    plt.xlabel('Trial #')
    for subplot in (211, 212):
        plt.subplot(subplot)
        dict = cumulative if subplot == 212 else simple
        for agent in new_agents:
            plt.plot(dict[agent][1:], range(1, num_trials + 1), label=agent.name)
    plt.legend()
    plt.show()
