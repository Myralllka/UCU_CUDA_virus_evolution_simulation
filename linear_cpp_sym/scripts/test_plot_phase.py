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

if __name__ == '__main__':
    create_plot()
