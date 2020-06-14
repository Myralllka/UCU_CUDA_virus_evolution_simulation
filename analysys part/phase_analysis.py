#!/usr/bin/python3
from sys import argv
import matplotlib.pyplot as plt
import statistics
from test_plot_phase import create_plot
import numpy


# host: 142.93.163.88
# port: 6006
# user: team1
# database: db1
# password: password1

# noinspection PyShadowingNames


def parse_file(f_name):
    """
    Parse file and return fore  arrays with dead, infected, patient and with immunity with deceleration
    of printed eras of the simulation
    :param f_name: file name of the file to parse.
    :return: [[general_people_number, immunity(list), infected(list), patient(list), isolated(list)], dead]
    """
    # result arrays
    isolated_a, infected_a, patient_a, immunity_a, dead_a = [], [], [], [], []

    with open(f_name, 'r', encoding='utf-8') as f:
        iter_step = int(f.readline().strip())
        people_num = int(f.readline().strip())
        for line in f:
            immunity, infected, patient, isolated, dead = line.strip().split()
            isolated_a.append(int(isolated))
            # Total infected
            infected_a.append(int(infected))
            patient_a.append(int(patient))
            immunity_a.append(int(immunity))
            dead_a.append(int(dead))

    return iter_step, people_num, immunity_a, infected_a, patient_a, isolated_a, dead_a


def plot_one_iter(iter_steps, snapshot_num, people_num, all_dict):
    """
    visualize the arrays
    """
    step_arr = [step_id for step_id in range(0, iter_steps * snapshot_num, iter_steps)]
    for key in all_dict:
        # print(type(all_dict[key]))
        # return 0
        plt.plot(step_arr, tuple(map(lambda n: n / people_num * 100, all_dict[key])), label=key)
        plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        plt.ylabel(key)
        plt.xlabel("era number")
        plt.grid(True)
        # plt.margins(x=-0.25, y=-0.25)
    plt.legend()
    plt.ylabel("% of dead people")
    plt.savefig('./plots/foo{}.png'.format(near_res_count - 5))
    plt.show()


def preprocessing_before_plot_one_iter(all_dict):
    shortest_length = min(len(all_dict[key]) for key in all_dict)
    print(shortest_length)

    for key in all_dict:
        all_dict[key] = all_dict[key][:shortest_length]

    return all_dict, shortest_length


def preprocessing_before_phase_plot(all_dict, profit=True):

    if profit:
        min_death = min(all_dict[key][-1] for key in all_dict)
        print("MIN DEATH", min_death)
        max_death = max(all_dict[key][-1] for key in all_dict)
        print("MAX DEATH", max_death)
        percent_from_all_death = min_death * 100 / max_death

        profit_death_percent = 100 - percent_from_all_death
        profit_death = max_death - min_death

        avg_death_gnd = min_death + int(profit_death * 0.33)
        high_death_gnd = avg_death_gnd + int(profit_death * 0.33)

    else:
        avg_death_gnd = 330000
        high_death_gnd = 660000

    deaths_per_same_rate = [all_dict[key][-1] for key in all_dict]

    print("AVG GND", avg_death_gnd)

    print(deaths_per_same_rate)
    count = 0
    for death in deaths_per_same_rate:
        count += 1
        if death < avg_death_gnd:
            phase_plot_dict[colors[0]][0].append(near_res_count - 5)
            phase_plot_dict[colors[0]][1].append((count - 1) * 40000)
            # print(count)

        elif death >= avg_death_gnd and death < high_death_gnd:
            phase_plot_dict[colors[1]][0].append(near_res_count - 5)
            phase_plot_dict[colors[1]][1].append((count - 1) * 40000)
            # print(count)

        elif death > high_death_gnd:
            phase_plot_dict[colors[2]][0].append(near_res_count - 5)
            phase_plot_dict[colors[2]][1].append((count - 1) * 40000)
            # print(count)


def find_survival_profit(dead_rate_dict):
    min_death = min(dead_rate_dict[key][-1] for key in dead_rate_dict)
    # print(min_death)
    mean_death = statistics.mean(dead_rate_dict[key][-1] for key in dead_rate_dict)

    max_death = max(dead_rate_dict[key][-1] for key in dead_rate_dict)
    # print(max_death)

    percent_from_all_death = min_death * 100 / max_death
    print("Death rate = {}%".format(near_res_count - 5),
          "\t{}% = Maximum profit of survival with beds increase".format(int(100 - percent_from_all_death)))


if __name__ == '__main__':
    near_res_count = 5

    phase_plot_dict = dict()
    # green - low level
    # blue - medium level
    # red - high level

    colors = ["green", "blue", "red"]

    for i in colors:
        phase_plot_dict[i] = [[], []]

    # 20
    for i in range(1):
        # 0
        near_res_bed_count = 0
        f_dir = "../RES/res{}/".format(near_res_count)
        near_res_count += 5

        # the same dead rate, n_beds changes
        dead_rate_n_beds = dict()

        # 26
        for j in range(26):
            f_name = f_dir + "res{}/snap1.txt".format(near_res_bed_count)

            iter_step, people_num, immunity_a, infected_a, patient_a, isolated_a, dead_a = parse_file(f_name)

            dead_rate_n_beds['beds_{}'.format(near_res_bed_count)] = dead_a  # param to visualise
            near_res_bed_count += 40000

        # plot the graph ########################################
        temp_tuple = preprocessing_before_plot_one_iter(dead_rate_n_beds)
        plot_one_iter(iter_step, temp_tuple[1], people_num, temp_tuple[0])

        # find profit ###########################################
        # find_survival_profit(dead_rate_n_beds)

        # statistics per one run of death_coef ##################
        # preprocessing_before_phase_plot(dead_rate_n_beds)

    # create_plot(phase_plot_dict)

        # break
