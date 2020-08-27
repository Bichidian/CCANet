import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from CCADatasets import whiteboard_data, gaussian_data, mnist_data


def get_nprandom(seed):
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    elif seed is None:
        return np.random
    else:
        raise ValueError


class CCAImplausible:
    def __init__(self, X: np.ndarray, Y: np.ndarray, k, eta, steps, seed=None):
        """Initialize CCA neural network.

        :param X: m by T dataset.
        :param Y: n by T dataset.
        :param k: number of neurons.
        :param eta: dictionary of learning rates, should has entries "W", "V", "M", "Lambda", "Gamma".
        :param seed: seed for generating random initial condition.
        """
        self.X = X
        self.Y = Y
        self.k = k
        self.eta = eta
        self.steps = steps
        self.obj = np.zeros([self.steps])
        self.err = np.zeros([self.steps])

        self.m = X.shape[0]
        self.n = Y.shape[0]
        self.T = X.shape[1]
        assert Y.shape[1] == self.T, "X.shape={}, Y.shape={}".format(X.shape, Y.shape)

        self.C_XX = self.X @ self.X.T / self.T
        self.C_YY = self.Y @ self.Y.T / self.T
        self.C_XY = self.X @ self.Y.T / self.T
        self.C_YX = self.C_XY.T

        self.W_opt, self.V_opt, self.obj_opt = self.opt_WVobj()
        # print("obj_opt: {}".format(self.obj_opt))

        nprandom = get_nprandom(seed)
        self.W = nprandom.randn(self.m, self.k)
        self.V = nprandom.randn(self.n, self.k)
        self.Lambda = nprandom.randn(self.k, self.k)
        self.Gamma = nprandom.randn(self.k, self.k)
        self.Lambda = (self.Lambda + self.Lambda.T) / 2
        self.Gamma = (self.Gamma + self.Gamma.T) / 2

    def reset(self, seed=None):
        """Reinitialize.

        :param seed: seed for generating random initial condition.
        """
        nprandom = get_nprandom(seed)
        self.W = nprandom.randn(self.m, self.k)
        self.V = nprandom.randn(self.n, self.k)
        self.Lambda = nprandom.randn(self.k, self.k)
        self.Gamma = nprandom.randn(self.k, self.k)
        self.Lambda = (self.Lambda + self.Lambda.T) / 2
        self.Gamma = (self.Gamma + self.Gamma.T) / 2

    def opt_WVobj(self):
        """Compute optimal W, V and objective.

        :return optimal W, V and objective.
        """
        C_XX_invsqrt = scipy.linalg.inv(scipy.linalg.sqrtm(self.C_XX))
        C_YY_invsqrt = scipy.linalg.inv(scipy.linalg.sqrtm(self.C_YY))
        K = C_XX_invsqrt @ self.C_XY @ C_YY_invsqrt
        W_white, S, Vh_white = np.linalg.svd(K)
        # print("S: {}".format(S))
        W_opt = C_XX_invsqrt @ W_white[:, 0: self.k]
        V_opt = C_YY_invsqrt @ Vh_white.T[:, 0: self.k]

        cov_WX_VY = np.trace(W_opt.T @ self.C_XY @ V_opt)
        var_WX = np.trace(W_opt.T @ self.C_XX @ W_opt)
        var_VY = np.trace(V_opt.T @ self.C_YY @ V_opt)
        obj_opt = cov_WX_VY / np.sqrt(var_WX * var_VY)

        return W_opt, V_opt, obj_opt

    def set_opt(self):
        """Set the network to optimal."""
        self.W = self.W_opt.copy()
        self.V = self.V_opt.copy()
        self.Lambda = self.W.T @ (self.C_XX @ self.W + self.C_XY @ self.V) * np.eye(self.k)
        self.Gamma = self.V.T @ (self.C_YX @ self.W + self.C_YY @ self.V) * np.eye(self.k)

        I_X = self.W.T @ self.X
        I_Y = self.V.T @ self.Y
        Z = I_X + I_Y
        assert np.allclose(self.X @ Z.T, self.X @ I_X.T @ self.Lambda)
        assert np.allclose(self.Y @ Z.T, self.Y @ I_Y.T @ self.Gamma)

    def perturb(self, noise, seed=None):
        """Perturb the network parameters.

        :param noise: noise level.
        :param seed: seed to generate random noise.
        """
        nprandom = get_nprandom(seed)
        self.W += nprandom.randn(self.m, self.k) * noise
        self.V += nprandom.randn(self.n, self.k) * noise
        self.Lambda += nprandom.randn(self.k, self.k) * noise
        self.Gamma += nprandom.randn(self.k, self.k) * noise
        self.Lambda = (self.Lambda + self.Lambda.T) / 2
        self.Gamma = (self.Gamma + self.Gamma.T) / 2

    def objective(self):
        """Compute the objective function.

        :return: objective.
        """
        cov_WX_VY = np.trace(self.W.T @ self.C_XY @ self.V)
        var_WX = np.trace(self.W.T @ self.C_XX @ self.W)
        var_VY = np.trace(self.V.T @ self.C_YY @ self.V)
        return cov_WX_VY / np.sqrt(var_WX * var_VY)

    def error_angle(self):
        """Compute the angle between current W, V and W_opt, V_opt under diag(C_XX, C_YY) norm.

        :return: error angle.
        """
        inner_WV_WVopt = np.trace(abs(self.W.T @ self.C_XX @ self.W_opt) + abs(self.V.T @ self.C_YY @ self.V_opt))
        inner_WV_WV = np.trace(self.W.T @ self.C_XX @ self.W + self.V.T @ self.C_YY @ self.V)
        inner_WVopt_WVopt = np.trace(self.W_opt.T @ self.C_XX @ self.W_opt + self.V_opt.T @ self.C_YY @ self.V_opt)
        cos_theta = inner_WV_WVopt / np.sqrt(inner_WV_WV * inner_WVopt_WVopt)
        theta = np.arccos(abs(cos_theta))
        return theta * 180 / np.pi

    def if_converged(self):
        """Calculate absolute and relative gradient to determine whether the algorithm has converged."""
        dW = self.C_XX @ self.W + self.C_XY @ self.V - self.C_XX @ self.W @ self.Lambda
        dV = self.C_YX @ self.W + self.C_YY @ self.V - self.C_YY @ self.V @ self.Gamma
        dLambda = self.W.T @ self.C_XX @ self.W - np.eye(self.k)
        dGamma = self.V.T @ self.C_YY @ self.V - np.eye(self.k)

        print("absolute gradient: {}".format(
            max(abs(dW).max(), abs(dV).max(), abs(dLambda).max(), abs(dGamma).max())))
        print("relative gradient: {}".format(
            max(abs(dW / self.W).max(), abs(dV / self.V).max(),
                abs(np.diag(dLambda) / np.diag(self.Lambda)).max(), abs(np.diag(dGamma) / np.diag(self.Gamma)).max())))

    def offline_train(self):
        """Offline training according to CCANet algorithm.

        :param steps: number of training iterations.
        """
        for t in range(self.steps):
            self.obj[t], self.err[t] = self.objective(), self.error_angle()
            self.offline_step()

        # print("W.T @ C_XX @ W:\n{}".format(self.W.T @ self.C_XX @ self.W))
        # print("W:\n{}".format(self.W))
        # print("W_opt:\n{}".format(self.W_opt))

    def offline_step(self):
        """Perform an offline step."""
        dW = self.C_XX @ self.W + self.C_XY @ self.V - self.C_XX @ self.W @ self.Lambda
        dV = self.C_YX @ self.W + self.C_YY @ self.V - self.C_YY @ self.V @ self.Gamma
        dLambda =  1/2 * self.W.T @ self.C_XX @ self.W - np.eye(self.k)
        dGamma =  1/2 * self.V.T @ self.C_YY @ self.V - np.eye(self.k)

        # I_X = self.W.T @ self.X
        # I_Y = self.V.T @ self.Y
        # Z = I_X + I_Y
        # dW2 = 1 / self.T * self.X @ (Z.T - I_X.T @ self.Lambda)
        # dV2 = 1 / self.T * self.Y @ (Z.T - I_Y.T @ self.Gamma)
        # dLambda2 = 1 / self.T * I_X @ I_X.T - np.eye(self.k)
        # dGamma2 = 1 / self.T * I_Y @ I_Y.T - np.eye(self.k)
        #
        # assert np.allclose(dW, dW2)
        # assert np.allclose(dV, dV2)
        # assert np.allclose(dLambda, dLambda2)
        # assert np.allclose(dGamma, dGamma2)

        self.W += self.eta["W"] * dW
        self.V += self.eta["V"] * dV
        self.Lambda += self.eta["Lambda"] * dLambda
        self.Gamma += self.eta["Gamma"] * dGamma

    def online_train(self, seed=None):
        """Online training according to CCANet algorithm.

        :param steps: number of training iterations.
        """
        nprandom = get_nprandom(seed)
        for t in range(self.steps):
            self.obj[t], self.err[t] = self.objective(), self.error_angle()
            self.online_step(nprandom.randint(self.T))

        # print("W.T @ C_XX @ W:\n{}".format(self.W.T @ self.C_XX @ self.W))

    def online_step(self, index):
        """Perform an online step.

        :param index: X[:, index] and Y[:, index] will be used for this step.
        """
        I_X = self.W.T @ self.X[:, [index]]
        I_Y = self.V.T @ self.Y[:, [index]]
        Z = I_X + I_Y
        self.W += self.eta["W"] * self.X[:, [index]] @ (Z.T - I_X.T @ self.Lambda)
        self.V += self.eta["V"] * self.Y[:, [index]] @ (Z.T - I_Y.T @ self.Gamma)
        self.Lambda += self.eta["Lambda"] * 1/2 * (I_X @ I_X.T - np.eye(self.k))
        self.Gamma += self.eta["Gamma"] * 1/2 * (I_Y @ I_Y.T - np.eye(self.k))


def main():
    m = 5
    n = 5
    k = 2
    T = 10000
    steps = 10000
    eta = {"W": 1e-3, "V": 1e-3, "M": 1e-2, "Lambda": 1e-2, "Gamma": 1e-2}
    X, Y = gaussian_data(m, n, T)
    n_run = 10

    implausible = list()
    obj_avg = 0
    for i in range(n_run):
        implausible.append(CCAImplausible(X, Y, k, eta, steps))
        implausible[i].online_train()
        obj_avg += implausible[i].obj / n_run
    plt.plot(obj_avg / implausible[0].obj_opt, label="implausible, {}".format(eta))
    plt.xlim([0, steps])
    plt.ylim([0, 1.2])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()