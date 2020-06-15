import numpy as np

np.random.seed(19680801)
import matplotlib.pyplot as plt



def create_plot(dicttt):

    fig, ax = plt.subplots()
    colors = ['green', 'blue']
    labels = ['low rate', 'middle rate', 'high rate']
    for i in range(len(colors)):
        x, y = dicttt[colors[i]]
        # scale = 200.0 * np.random.rand(4)
        # scale = [10 for i in range()]
        ax.scatter(x, y, c=colors[i], label=labels[i], edgecolors='none')

    ax.legend()

    plt.ylabel("% of isolation places provided")
    plt.xlabel("% of death rate")

    ax.grid(True)
    plt.savefig('./analysis_results/immunity_beds_best_profit_rate.png')
    plt.show()


def test_phase_difference(xyz, beds_1, beds_2):
    # xyz = np.array(np.random.random((3, 3)))
    # print(xyz)
    # print(xyz[:, 0])
    marker_size = 30
    plt.scatter(xyz[0], xyz[1], marker_size, c=xyz[2])
    plt.title("{} vs {} beds".format(beds_1, beds_2))
    plt.xlabel("Epoch")
    plt.ylabel("Max difference")
    cbar = plt.colorbar()
    plt.grid(True)
    cbar.set_label("death_rate", labelpad=+1)
    plt.show()


def test_phase_integral(xyz):
    # xyz = np.array(np.random.random((3, 3)))
    # print(xyz)
    # print(xyz[:, 0])
    marker_size = 2000
    plt.scatter(xyz[0], xyz[1], marker_size, c=xyz[2], marker="s")
    plt.title("Integrals between isolation places")
    plt.xlabel("% Death rate")
    plt.ylabel("% Isolation places")
    cbar = plt.colorbar()
    plt.grid(True)
    cbar.set_label("Integral size", labelpad=+1)
    plt.show()

if __name__ == '__main__':
    # create_plot()
    test_phase_difference('a')
