import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pickle
from time import time
from CCADatasets import gaussian_data, mnist_data, mediamill_data
from CCAPyramidal import CCAPyramidal
from CCAImplausible import CCAImplausible
from CCASrebro import CCASrebro


def lighted(color):
    return 1 / 3 * color + 2 / 3 * np.array([1, 1, 1])


class CCAComparer:
    def __init__(self, X, Y, k, steps, n_run, eta_pyramidal, alpha_pyramidal, eta_srebro, tau_srebro):
        """Initialize an object that compares the performance of proposed, nonlocal, nondecay and MSG-CCA algorithms.

        :param X: m by T dataset.
        :param Y: n by T dataset.
        :param k: number of canonical components to extract, which is also the number of neurons.
        :param steps: number of training steps.
        :param n_run: each algorithm is run repeatedly for this many times; but MSG-CCA is only run once.
        :param eta_pyramidal: learning rates for proposed, nonlocal and nondecay algorithms.
        :param alpha_pyramidal: learning rate decay rate for proposed algorithm.
        :param eta_srebro: the eta parameter for MSG-CCA algorithm.
        :param tau_srebro: the tau parameter for MSG-CCA algorithm.
        """
        self.n_run = n_run
        self.steps = steps
        self.pyramidal = [CCAPyramidal(X, Y, k, eta_pyramidal, steps, alpha_pyramidal, mode="hierarchy") for _ in range(n_run)]
        self.nondecay = [CCAPyramidal(X, Y, k, eta_pyramidal, steps, alpha=0, mode="hierarchy") for _ in range(n_run)]
        self.implausible = [CCAImplausible(X, Y, k, eta_pyramidal, steps) for _ in range(n_run)]
        self.srebro = CCASrebro(X, Y, k, eta_srebro, tau_srebro, steps)

    def run(self):
        print("proposed:")
        for i in range(self.n_run):
            self.pyramidal[i].online_train()
            print(i)
        print("nondecay:")
        for i in range(self.n_run):
            self.nondecay[i].online_train()
            print(i)
        print("nonlocal:")
        for i in range(self.n_run):
            self.implausible[i].online_train()
            print(i)
        print("MSG-CCA:")
        self.srebro.train()
        print(0)

    def save(self, filename):
        pickle.dump(self, open(filename, mode="xb"))

    def relative_obj(self, neuron):
        return neuron.obj / neuron.obj_opt

    def show_one(self, neurons, color, ax1, ax2, sigma, label):
        relobj = np.zeros([self.n_run, self.steps])
        for i in range(self.n_run):
            relobj[i, :] = self.relative_obj(neurons[i])
        ax1.plot(gaussian_filter(relobj.mean(axis=0), sigma=sigma), c=color, linewidth=1, alpha=0.8, label=label)
        ax1.fill_between(x=np.arange(self.steps), y1=gaussian_filter(relobj.mean(axis=0) + relobj.std(axis=0), sigma=sigma),
                         y2=gaussian_filter(relobj.mean(axis=0) - relobj.std(axis=0), sigma=sigma), color=lighted(color), alpha=0.2)

        err = np.zeros([self.n_run, self.steps])
        for i in range(self.n_run):
            err[i, :] = neurons[i].err
        ax2.plot(gaussian_filter(err.mean(axis=0), sigma=sigma), c=color, linewidth=1, alpha=0.8, label=label)
        ax2.fill_between(x=np.arange(self.steps), y1=gaussian_filter(err.mean(axis=0) + err.std(axis=0), sigma=sigma),
                         y2=gaussian_filter(err.mean(axis=0) - err.std(axis=0), sigma=sigma), color=lighted(color), alpha=0.2)

    def show_all(self, color, ax1, ax2, label_a="a", label_b="b", sigma=1000):
        for i in range(3):
            neurons = [self.implausible, self.nondecay, self.pyramidal][i]
            self.show_one(neurons, color[i], ax1, ax2, sigma, label=["nonlocal", "nondecay", "proposed"][i])
        ax1.plot(self.relative_obj(self.srebro), c=color[3], linewidth=1, label="MSG-CCA")

        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend([handles[i] for i in [2, 0, 1, 3]], [labels[i] for i in [2, 0, 1, 3]])
        ax1.set_xlim([0, self.steps])
        ax1.set_ylim([0, 1.2])
        ax1.set_ylabel("Normalized objective")
        ax1.text(-0.12, 1, label_a,
                 horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes, fontweight="bold")

        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend([handles[i] for i in [2, 0, 1]], [labels[i] for i in [2, 0, 1]])
        ax2.set_xlim([0, self.steps])
        ax2.set_ylim([0, 90])
        ax2.set_ylabel("Angular error")
        ax2.text(-0.12, 1, label_b,
                 horizontalalignment="center", verticalalignment="center", transform=ax2.transAxes, fontweight="bold")


def main():
    m = 5
    n = 5
    k = 3
    T = 10000
    steps = 300000
    eta_pyramidal = {"W": 0.02, "V": 0.02, "M": 0.02, "Lambda": 0.02, "Gamma": 0.02}
    alpha_pyramidal = 5e-6
    eta_srebro = 0.1
    tau_srebro = 100
    n_run = 10

    X1, Y1 = gaussian_data(m, n, T)
    X2, Y2 = mnist_data(m, n, T)
    X3, Y3 = mediamill_data(m, n, T)
    X = [X1, X2, X3]
    Y = [Y1, Y2, Y3]
    dataset = ["gaussian", "mnist", "mediamill"]

    for i in range(3):
        comparer = CCAComparer(X[i], Y[i], k, steps, n_run, eta_pyramidal, alpha_pyramidal, eta_srebro, tau_srebro)
        comparer.run()
        comparer.save("./result_data/compare_{}.pickle".format(dataset[i]))


if __name__ == "__main__":
    main()  # took 11657 seconds to run
