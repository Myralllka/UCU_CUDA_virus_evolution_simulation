#!/usr/bin/python3
from sys import argv

import matplotlib.pyplot as plt


# noinspection PyShadowingNames
def parse_file(f_name):
    """
    Parse file and return fore  arrays with dead, infected, patient and with immunity with deceleration
    of printed eras of the simulation
    :param f_name: file name of the file to parse.
    :return: [[normal(list), infected(list), patient(list), immunity(list), dead(list)], iter_step]
    """
    # result arrays
    normal_a, infected_a, patient_a, immunity_a, dead_a = [], [], [], [], []

    with open(f_name, 'r', encoding='utf-8') as f:
        iter_step = int(f.readline().strip())
        for line in f:
            normal, infected, patient, immunity, _isolated, dead = line.strip().split()
            normal_a.append(int(normal))
            infected_a.append(int(infected))
            patient_a.append(int(patient))
            immunity_a.append(int(immunity))
            dead_a.append(int(dead))

    return iter_step, normal_a, infected_a, patient_a, immunity_a, dead_a


def plot_one_iter(iter_steps, snapshot_num, **kwargs):
    """
    visualize the arrays
    """
    step_arr = [step_id for step_id in range(0, iter_steps * snapshot_num, iter_steps)]
    for key in kwargs:
        plt.plot(step_arr, kwargs[key], label=key)
        plt.ylabel(key)
        plt.xlabel("era number")
        plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    if len(argv) == 1:
        f_name = "../res/snap.txt"
    elif len(argv) == 2:
        f_name = argv[1]
    else:
        raise Exception("Invalid params")

    iter_step, normal_a, infected_a, patient_a, immunity_a, dead_a = parse_file(f_name)
    plot_one_iter(iter_step, len(normal_a),
                  normal_a=normal_a, infected_a=infected_a, patient_a=patient_a, immunity_a=immunity_a, dead_a=dead_a)

    # digits in sub plot [num of rows, num of cols, index]
    # subplot(221)
    # subplot(222)
    # subplot(223)
    # subplot(224)

    # tight_layout()  # subplots collecting
    # show()
