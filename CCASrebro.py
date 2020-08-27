import numpy as np
import scipy.linalg
import matlab.engine

eng = matlab.engine.start_matlab()


def get_nprandom(seed):
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    elif seed is None:
        return np.random
    else:
        raise ValueError


class CCASrebro:
    def __init__(self, X, Y, k, eta, tau, steps):
        """Initialize an MSG-CCA algorithm.

        :param X: m by T dataset.
        :param Y: n by T dataset.
        :param k: number of canonical components to extract.
        :param eta: learning rate.
        :param tau: number of time steps that are used to initially estimate covariance matrix.
        :param steps: number of training steps.
        """
        self.X = X
        self.Y = Y
        self.k = k
        self.eta = eta
        self.tau = tau
        self.steps = steps
        self.obj = np.zeros([self.steps])

        self.d_x = X.shape[0]
        self.d_y = Y.shape[0]
        self.T = X.shape[1]
        assert Y.shape[1] == self.T

        self.C_xx = self.X @ self.X.T / self.T
        self.C_yy = self.Y @ self.Y.T / self.T
        self.C_xy = self.X @ self.Y.T / self.T
        self.C_yx = self.C_xy.T
        self.C_xy_white = scipy.linalg.inv(scipy.linalg.sqrtm(self.C_xx)) @ self.C_xy \
                          @ scipy.linalg.inv(scipy.linalg.sqrtm(self.C_yy))

        self.U_opt, self.V_opt, self.obj_opt = self.opt_WVobj()

        self.C_xx_est = X[0:self.tau] @ X[0:self.tau].T / self.tau
        self.C_yy_est = Y[0:self.tau] @ Y[0:self.tau].T / self.tau
        self.W_x = scipy.linalg.inv(scipy.linalg.sqrtm(self.C_xx_est))
        self.W_y = scipy.linalg.inv(scipy.linalg.sqrtm(self.C_yy_est))

        self.M = np.zeros([self.d_x, self.d_y])

    def opt_WVobj(self):
        """Compute optimal W, V and objective.

        :return optimal W, V and objective.
        """
        C_XX_invsqrt = scipy.linalg.inv(scipy.linalg.sqrtm(self.C_xx))
        C_YY_invsqrt = scipy.linalg.inv(scipy.linalg.sqrtm(self.C_yy))
        K = C_XX_invsqrt @ self.C_xy @ C_YY_invsqrt
        W_white, S, Vh_white = np.linalg.svd(K)
        W_opt = C_XX_invsqrt @ W_white[:, 0: self.k]
        V_opt = C_YY_invsqrt @ Vh_white.T[:, 0: self.k]

        cov_WX_VY = np.trace(W_opt.T @ self.C_xy @ V_opt)
        var_WX = np.trace(W_opt.T @ self.C_xx @ W_opt)
        var_VY = np.trace(V_opt.T @ self.C_yy @ V_opt)
        obj_opt = cov_WX_VY / np.sqrt(var_WX * var_VY)

        return W_white[:, 0: self.k], Vh_white.T[:, 0: self.k], obj_opt

    def set_opt(self):
        self.M = self.U_opt @ self.V_opt.T

    def objective(self):
        return np.trace(self.M.T @ self.C_xy_white) / self.k

    def projection(self):
        """Projection of the M matrix. This function calls an external matlab script, written by the authors of the
        original paper."""
        U, S, Vh = scipy.linalg.svd(self.M)
        S = np.array(eng.yet_another_projection(matlab.double(list(S)), matlab.double([self.k]))[0])
        # print(S, S.sum())
        sigma = np.zeros([self.d_x, self.d_y])
        for i in range(min(self.d_x, self.d_y)):
            sigma[i, i] = S[i]
        self.M = U @ sigma @ Vh

    def decay(self, t):
        """Return learning rate decay factor."""
        return (t + 1) ** -0.5

    def train(self):
        """Train the algorithm."""
        for t in range(self.steps):
            self.obj[t] = self.objective()
            self.step(t, np.random.randint(self.T))

    def step(self, t, index):
        """A step of the MSG-CCA algorithm."""
        self.C_xx_est = (t + self.tau - 1) / (t + self.tau) * self.C_xx_est \
                        + 1 / (t + self.tau) * self.X[:, [index]] @ self.X[:, [index]].T
        self.C_yy_est = (t + self.tau - 1) / (t + self.tau) * self.C_yy_est \
                        + 1 / (t + self.tau) * self.Y[:, [index]] @ self.Y[:, [index]].T
        self.W_x = scipy.linalg.inv(scipy.linalg.sqrtm(self.C_xx_est))
        self.W_y = scipy.linalg.inv(scipy.linalg.sqrtm(self.C_yy_est))
        self.M += self.eta * self.decay(t) * self.W_x @ self.X[:, [index]] @ self.Y[:, [index]].T @ self.W_y
        self.projection()
