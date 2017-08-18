import numpy as np
import matplotlib.pyplot as plt


def generate_regret_curves(bandits, pull_max, slot_machines, num_trials=50):
    """Generate and graph regret curves for bandits x pull_values on simulator.

    :param bandits: the bandit classes to use.
    :param pull_max: the maximum number of pulls to use.
    :param slot_machines: slot machines (BanditProblems) from vegas which will be used to measure regret.
    :param num_trials: for how many trials we will run.
    """
    # Generate permutations of bandits using pull values
    num_increments = 10
    pull_increments = range(int(pull_max/num_increments),
                            int(pull_max + pull_max/num_increments),
                            int(pull_max/num_increments))

    # For each agent, initialize average regret
    cumulative, simple = {}, {}
    for bandit in bandits:  # for each agent category
        cumulative[bandit], simple[bandit] = [[[0 for _ in range(num_increments)]
                                               for _ in range(len(slot_machines))]
                                              for _ in range(2)]

    # Generate data
    for bandit in bandits:
        for machine_idx, machine in enumerate(slot_machines):
            for pull_idx, num_pulls in enumerate(pull_increments):  # different num_pulls configurations
                new_bandit = bandit(machine.num_arms)
                for trial_num in range(1, num_trials + 1):
                    # Run trial
                    run_trial(new_bandit, machine, num_pulls)

                    # Calculate regrets
                    c, s = get_cumulative_regret(new_bandit, machine.max_expected_reward), \
                           get_simple_regret(new_bandit, machine.max_expected_reward)

                    # Update our averages
                    cumulative[bandit][machine_idx][pull_idx] = \
                        update_rolling_avg(cumulative[bandit][machine_idx][pull_idx], trial_num, c)
                    simple[bandit][machine_idx][pull_idx] = \
                        update_rolling_avg(simple[bandit][machine_idx][pull_idx], trial_num, s)

    for bandit in bandits:
        # Construct and display graphs for each bandit algorithm
        plt.figure()
        for regret_type in ('Cumulative', 'Simple'):
            ax = plt.subplot(121 if regret_type == 'Cumulative' else 122)
            ax.grid('on')

            ax.set_xlim([pull_increments[0], pull_increments[-1]])
            ax.set_xlabel('Number of Pulls')
            ax.set_ylabel(regret_type + ' Regret')

            regret_dict = cumulative if regret_type == 'Cumulative' else simple

            for machine_idx in range(len(slot_machines)):
                ax.plot(pull_increments, regret_dict[bandit][machine_idx], label='Slot machine ' + str(machine_idx + 1))

            ax.legend(loc='lower right')
            ax.set_title(regret_type + ' Regret for {} ({} trials)'.format(bandit.name, num_trials))

    plt.show()


def update_rolling_avg(current_avg, trial_num, new_value):
    return (current_avg * (trial_num - 1) + new_value) / trial_num


def run_trial(bandit, machine, num_pulls):
    bandit.initialize()
    for pull in range(num_pulls):
        arm = bandit.select_pull_arm()
        reward = machine.pull(arm)
        bandit.update(arm, reward)


def get_cumulative_regret(bandit, max_reward):
    """Return the accrued cumulative regret."""
    return bandit.total_pulls * max_reward - np.sum(bandit.average_reward * bandit.num_pulls)


def get_simple_regret(bandit, max_reward):
    return max_reward - bandit.get_best_reward()
