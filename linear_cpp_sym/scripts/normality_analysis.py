from sys import argv
from scipy import stats
import numpy as np
from functools import reduce


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
    count_lines = 0

    with open(f_name, 'r', encoding='utf-8') as f:
        iter_step = int(f.readline().strip())
        people_num = int(f.readline().strip())
        for line in f:
            cur_line = line.strip().split()
            if len(cur_line) == 5:
                immunity, infected, patient, isolated, dead = line.strip().split()

                # if not ((count_lines > 100) and ((infected == patient) and (patient == isolated))):
                isolated_a.append(int(isolated))
                # Total infected
                infected_a.append(int(infected))
                patient_a.append(int(patient) + int(isolated))
                immunity_a.append(int(immunity))
                dead_a.append(int(dead))

    return iter_step, people_num, immunity_a, infected_a, patient_a, isolated_a, dead_a


def dict_of_files(num_of_files):
    """
    Create dictionary where [keys] - numbers of parsed file, [value] - list of lists with corresponding to file
    data for different statistics (from naming).
    """
    all_statistics = dict()
    for i in range(1, num_of_files + 1):
        f_name = "../res/snap%i.txt" % i
        all_statistics[i] = parse_file(f_name)
    return all_statistics


def data_filtering(all_statistics_dict: dict, gap):
    """
    Return dictionary with filtered data by the gap (each [gap]th recording from file)
    with [key] - statistic (from naming), [value] - gathered data from 'raw' dictionary.
    """
    result = dict()
    for i in all_statistics_dict:
        for prop in range(2, 7):
            if naming[prop] not in result.keys():
                result[naming[prop]] = []
            result[naming[prop]].append(all_statistics_dict[i][prop][gap])

    return result


def normality_test(filtered_dict):
    """
    Tests the sequence with 2-sided chi squared test and write to the [tests_outcome]
    n - if normally distributed
    . - if not
    """
    print("\nGAP = ", tmp_gap)

    for prop in naming[2:]:

        if prop not in tests_outcome.keys():
            tests_outcome[prop] = ''

        if prop not in variances_outcome.keys():
            variances_outcome[prop] = []

        data = filtered_dict[prop]
        k2, p = stats.normaltest(data)
        variances_outcome[prop].append(np.var(data))

        # print("----------\nTest for {} :".format(prop))
        # print("p = {:g}".format(p))
        if p < alpha:  # null hypothesis: x comes from a normal distribution
            # print("The null hypothesis can be rejected")
            tests_outcome[prop] += '.'
        else:
            # print("The null hypothesis cannot be rejected - normal distribution")
            tests_outcome[prop] += 'n'


if __name__ == '__main__':
    naming = ["iter_step", "people_num", "immunity_a", "infected_a", "patient_a", "isolated_a", "dead_a"]

    lines_to_work = 1200  # number of lines in files to analyse
    num_of_files = 50     # number of files to process
    gap = 400             # each [gap] lines to process
    tmp_gap = 0
    alpha = 0.05          # significance level of test

    number_of_tests = lines_to_work // gap
    tests_outcome = dict()

    variances_outcome = dict()

    all_dict = dict_of_files(num_of_files)
    for i in range(number_of_tests):
        tmp_gap += gap
        filtered = data_filtering(all_dict, tmp_gap)
        normality_test(filtered)

    print("\nSignificance level of test = {}, gap = {}, num of files = {}\n----------".format(alpha, gap, num_of_files))
    for prop in tests_outcome:
        if prop == 'dead_a':
            print(prop, '\t\t', tests_outcome[prop])
        else:
            print(prop, '\t', tests_outcome[prop])

    print('Variance')
    for vars in variances_outcome:
        if vars == 'dead_a':
            print(vars, '\t\t', variances_outcome[vars])

        else:
            print(vars, '\t', variances_outcome[vars])
