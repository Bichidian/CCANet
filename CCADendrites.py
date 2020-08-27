import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pickle
from tensorflow.examples.tutorials.mnist import input_data

# Automatically create folder and download data for the first time; reuse later
mnist = input_data.read_data_sets("./mnist/", reshape=False)


def get_nprandom(seed):
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    elif seed is None:
        return np.random
    else:
        raise ValueError


class CCADendrites:
    def __init__(self, X, w_opt, eta_w, eta_lambda, seed=None):
        """Initialize CCA neural network with multiple dendrites.

        :param X: a list of dataset matrices, the i_th of which has shape n_i by T.
        :param w_opt: the optimal weight.
        :param eta_w: learning rate of w.
        :param eta_lambda: learning rate of lambda.
        :param seed: seed for generating random initial condition.
        """
        self.X = X
        self.w_opt = w_opt
        self.eta_w = eta_w
        self.eta_lambda = eta_lambda

        self.d = len(self.X)
        self.n = np.zeros([self.d], dtype=int)
        self.T = self.X[0].shape[1]
        for i in range(self.d):
            self.n[i] = self.X[i].shape[0]
            assert self.T == self.X[i].shape[1]
        self.cov = self.calc_cov()

        nprandom = get_nprandom(seed)
        assert len(self.w_opt) == self.d
        self.w = list()
        for i in range(self.d):
            self.w.append(nprandom.randn(self.n[i], 1))
            assert self.w[i].shape == self.w_opt[i].shape
        self.lambd = list(nprandom.randn(self.d))

    def calc_cov(self):
        """Calculate covariance matrices among datasets.

        :return: a list of lists, cov[i][j] is covariance matrix between datasets X[i] and X[j].
        """
        cov = list()
        for i in range(self.d):
            cov.append(list())
            for j in range(self.d):
                if j < i:
                    cov[i].append(cov[j][i].T)
                else:
                    cov[i].append(1 / self.T * self.X[i] @ self.X[j].T)
                assert cov[i][j].shape == (self.n[i], self.n[j])
        return cov

    def reset(self, seed=None):
        """Set the network back to a random initial state.

        :param seed: seed for generating random initial condition.
        """
        nprandom = get_nprandom(seed)
        self.w = list()
        for i in range(self.d):
            self.w.append(nprandom.randn(self.n[i], 1))
        self.lambd = list(nprandom.randn(self.d))

    def objective(self):
        """Calculate the SUMCOR CCA objective function."""
        corr_sum = 0
        cnt = 0
        for i in range(self.d):
            for j in range(i + 1, self.d):
                cnt += 1
                corr_sum += self.w[i].T @ self.cov[i][j] @ self.w[j] / np.sqrt(
                    (self.w[i].T @ self.cov[i][i] @ self.w[i]) * (self.w[j].T @ self.cov[j][j] @ self.w[j]))
        assert cnt == self.d * (self.d - 1) / 2
        return corr_sum / cnt

    def error_angle(self):
        """Calculate the error angle, defined as the angle between current network weight and optimal weight,
        under a special inner product definition."""
        w_wopt_inner = 0
        w_w_inner = 0
        wopt_wopt_inner = 0
        for i in range(self.d):
            w_wopt_inner += abs(self.w[i].T @ self.cov[i][i] @ self.w_opt[i])
            w_w_inner += self.w[i].T @ self.cov[i][i] @ self.w[i]
            wopt_wopt_inner += self.w_opt[i].T @ self.cov[i][i] @ self.w_opt[i]
        cos_theta = w_wopt_inner / np.sqrt(w_w_inner * wopt_wopt_inner)
        theta = np.arccos(abs(cos_theta))
        return theta * 180 / np.pi

    def online_train(self, steps, seed=None):
        """Train the network in an online manner.

        :param steps: number of training steps.
        :param seed: random seed for sampling dataset timestep.
        """
        nprandom = get_nprandom(seed)
        self.obj = np.zeros([steps])
        self.err = np.zeros([steps])
        for t in range(steps):
            self.online_step(nprandom.randint(self.T))
            self.obj[t], self.err[t] = self.objective(), self.error_angle()

    def online_step(self, index):
        """Perform an online training step.

        :param index: the dataset timestep.
        """
        I = list()
        y = 0
        for i in range(self.d):
            I.append(self.w[i].T @ self.X[i][:, [index]])
            y += I[i]
        for i in range(self.d):
            self.w[i] += self.eta_w * self.X[i][:, [index]] * (y - self.lambd[i] * I[i])
            self.lambd[i] += 1/2 * self.eta_lambda * (I[i] * I[i] - 1)

    def offline_train(self, steps):
        """Train the network in an offline manner.

        :param steps: number of training steps.
        """
        self.obj = np.zeros([steps])
        self.err = np.zeros([steps])
        for t in range(steps):
            self.offline_step()
            self.obj[t], self.err[t] = self.objective(), self.error_angle()

    def offline_step(self):
        """Perform an offline training step."""
        w_new = self.w
        for i in range(self.d):
            sumj_Cij_wj = np.zeros([self.n[i], 1])
            for j in range(self.d):
                sumj_Cij_wj += self.cov[i][j] @ self.w[j]
            w_new[i] += self.eta_w * (sumj_Cij_wj - self.lambd[i] * self.cov[i][i] @ self.w[i])
            self.lambd[i] += 1/2 * self.eta_lambda * (self.w[i].T @ self.cov[i][i] @ self.w[i] - 1)
        self.w = w_new


class CCADendritesOnOff:
    def __init__(self, X, w_opt, eta_w, eta_lambda, steps):
        """Initialize CCA neural network with multiple dendrites, based on the CCADendrites class. This class trains
        the network twice, once online and once offline, in order to estimate error angle.

        :param X: a list of dataset matrices, the i_th of which has shape n_i by T.
        :param w_opt: this argument is useless, but you must pass it.
        :param eta_w: learning rate of w.
        :param eta_lambda: learning rate of lambda.
        :param steps: number of training steps.
        """
        self.X = X
        self.w_opt = w_opt
        self.eta_w = eta_w
        self.eta_lambda = eta_lambda
        self.steps = steps

    def run(self):
        """Train the network offline to obtain the optimal weight w_opt, then train the network online."""
        neuron = CCADendrites(self.X, self.w_opt, self.eta_w, self.eta_lambda)
        neuron.offline_train(self.steps)
        self.offline_obj = neuron.obj
        w_opt = neuron.w.copy()
        neuron.reset()
        neuron.w_opt = w_opt
        neuron.online_train(self.steps)
        self.online_obj = neuron.obj
        self.err = neuron.err
        self.w = neuron.w

    def show(self, ax1=plt.subplot(121), ax2=plt.subplot(122)):
        ax1.set_title("Objective")
        ax2.set_title("Error angle")
        ax1.set_xlim([0, self.steps])
        ax1.set_ylim([0, 1])
        ax2.set_xlim([0, self.steps])
        ax2.set_ylim([0, 90])

        ax1.plot(self.offline_obj, label="offline")
        ax1.plot(self.online_obj, label="online")
        ax2.plot(self.err)
        ax1.legend()
        plt.show()

    def save(self, filename):
        pickle.dump(self, open(filename, mode="xb"))


def whiteboard_data(n, T, seed=None):
    """Generate svd-max dataset, which is a dataset where all pairs of correlation can reach the maximum value 1.

    :param n: list of dimensionalities of all datasets.
    :param T: number of time steps in a dataset.
    :param seed: random seed.
    :return: a list of dataset matrices, the i_th of which has a shape of n[i] by T; and a list of optimal weights that
    should be reached by CCA algorithm.
    """
    nprandom = get_nprandom(seed)
    d = len(n)
    U = list()
    V = list()
    S = list()
    X = list()
    V_concat = scipy.linalg.orth(nprandom.randn(T, sum(n) - d + 1))
    for i in range(d):
        U.append(scipy.linalg.orth(nprandom.randn(n[i], n[i])))
        V.append(V_concat[:, (sum(n[0:i]) - i): (sum(n[0:(i+1)]) - i)])
        V[i][:, 0] = V_concat[:, 0]
        S.append(np.diag(nprandom.uniform(0.1, 1, size=n[i])))
        X.append(U[i] @ S[i] @ V[i].T)
        X[i] *= np.sqrt(T)

    w_opt = list()
    for i in range(d):
        w_opt.append(1 / S[i][0, 0] * U[i][:, [0]])
        assert w_opt[i].shape == (n[i], 1)

    return X, w_opt


def mnist_4X(n_sqrt, T):
    """Generate mnist4X dataset, in which the MNIST image is split into four quarters.

    :param n_sqrt: the side length of the four square patches that meet at the center of MNIST image. Maximum 14.
    :param T: number of time steps in a dataset. Maxmimum is number of MNIST images.
    :return: a list of four dataset matrices, all of which have a shape of n_sqrt**2 by T; and a list of dummy weight
    matrices for feeding the w_opt input argument.
    """
    mnist_less = mnist.train.images[0:T]
    X = list()
    X.append(mnist_less[:, (14 - n_sqrt):14, (14 - n_sqrt):14].reshape(T, -1).T)  # Left-up
    X.append(mnist_less[:, (14 - n_sqrt):14, 14:(14 + n_sqrt)].reshape(T, -1).T)  # Right-up
    X.append(mnist_less[:, 14:(14 + n_sqrt), (14 - n_sqrt):14].reshape(T, -1).T)  # Left-down
    X.append(mnist_less[:, 14:(14 + n_sqrt), 14:(14 + n_sqrt)].reshape(T, -1).T)  # Right-down
    for i in range(4):
        X[i] += np.random.randn(n_sqrt * n_sqrt, T) * 0.01
        X[i] -= X[i].mean(1, keepdims=True)

    w_opt = list()
    for i in range(4):
        w_opt.append(np.random.randn(n_sqrt * n_sqrt, 1))
        w_opt[i] /= np.sqrt(w_opt[i].T @ X[i] @ X[i].T @ w_opt[i] / T)

    return X, w_opt


def show_mnist_weight(neuron):
    n_sqrt = int(np.sqrt(neuron.w[0].shape[0]))
    T = neuron.X[0].shape[1]
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        variances = np.zeros([n_sqrt * n_sqrt])
        for j in range(n_sqrt * n_sqrt):
            variances[j] = neuron.w[i][j] * neuron.X[i][[j]] @ neuron.X[i][[j]].T * neuron.w[i][j] / T
        plt.imshow(variances.reshape(n_sqrt, n_sqrt))
        plt.colorbar()
    plt.show()


def show_whiteboard_training(ax1=None, ax2=None):
    if ax1 is None:
        ax1 = plt.subplot(121)
    if ax2 is None:
        ax2 = plt.subplot(122)
    eta_w = 0.005
    eta_lambda = 0.005
    steps = 20000
    X, w_opt = whiteboard_data(n=[4, 5, 6], T=10000)

    neuron = CCADendrites(X, w_opt, eta_w, eta_lambda)
    neuron.online_train(steps)

    ax1.set_ylabel("Absolute objective")
    ax1.set_xlabel("Time steps")
    ax1.set_xlim([0, steps])
    ax1.set_ylim([0, 1])
    ax1.plot(neuron.obj)
    ax1.text(-0.12, 1, "e",
             horizontalalignment="center", verticalalignment="center", transform=ax1.transAxes, fontweight="bold")

    ax2.set_ylabel("Angular error")
    ax2.set_xlabel("Time steps")
    ax2.set_xlim([0, steps])
    ax2.set_ylim([0, 90])

    ax2.plot(neuron.err)
    ax2.text(-0.1, 1, "f",
             horizontalalignment="center", verticalalignment="center", transform=ax2.transAxes, fontweight="bold")

    plt.show()


def main():
    eta_w = 0.005
    eta_lambda = 0.005
    steps = 300000

    X, w_opt = mnist_4X(n_sqrt=14, T=10000)  # This w_opt is fake.
    onoff = CCADendritesOnOff(X, w_opt, eta_w, eta_lambda, steps)
    onoff.run()
    onoff.save("./result_data/dendrites_onoff_mnist4X.pickle")


if __name__ == "__main__":
    main()
