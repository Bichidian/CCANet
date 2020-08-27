import numpy as np
import scipy.linalg
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import load_svmlight_file
import os

# Automatically create folder and download data for the first time; reuse later
mnist = input_data.read_data_sets("./mnist/", reshape=False)


def get_nprandom(seed):
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    elif seed is None:
        return np.random
    else:
        raise ValueError


def whiteboard_data(m, n, k, T):
    """Generate svd-max dataset, which is a dataset where all pairs of correlation can reach the maximum value 1.

    :param m: dimensionality of first dataset.
    :param n: dimensionality of second dataset.
    :param k: number of identical dimensions (before rotation).
    :param T: number of time steps in a dataset.
    :return: a tuple of two dataset matrices, whose shapes are m by T and n by T.
    """
    U_X = scipy.linalg.orth(np.random.randn(m, m))
    U_Y = scipy.linalg.orth(np.random.randn(n, n))
    V_XandY = scipy.linalg.orth(np.random.randn(T, m + n - k))
    V_X = V_XandY[:, :m]
    V_Y = V_XandY[:, (m - k):]
    assert V_X.shape == (T, m) and V_Y.shape == (T, n), "{}{}".format(V_X.shape, V_Y.shape)
    S_X = np.diag(np.random.randn(m))
    S_Y = np.diag(np.random.randn(n))

    X = U_X @ S_X @ V_X.T
    Y = U_Y @ S_Y @ V_Y.T
    X -= X.mean(axis=1, keepdims=True)  # Mean-centering
    Y -= Y.mean(axis=1, keepdims=True)  # Mean-centering
    X *= np.sqrt(T)
    Y *= np.sqrt(T)

    return X, Y


def gaussian_data(m, n, T, seed=None):
    """Generate Gaussian dataset with random covariance matrix.

    :param m: dimensionality of first dataset.
    :param n: dimensionality of second dataset.
    :param T: number of time steps in a dataset.
    :param seed: random seed.
    :return: a tuple of two dataset matrices, whose shapes are m by T and n by T.
    """
    nprandom = get_nprandom(seed)
    mean = np.zeros([m + n])
    cov = nprandom.randn(m + n, m + n)
    cov = cov @ cov.T / (m + n) / 10
    X_Y = nprandom.multivariate_normal(mean, cov, size=T).T
    X = X_Y[0:m]
    Y = X_Y[m:(m + n)]
    return X, Y


def mnist_data(m, n, T):
    """Extract MNIST data from files.

    :param m: Dimensionality of left half of 15th row, maximum 14.
    :param n: Dimensionality of right half of 15th row, maximum 14.
    :param T: Number of frames, maximum 55000.
    :return: part of left and right half of 15th row.
    """
    X = mnist.train.images[0:T, 14, (14-m):14].reshape(T, m).T  # Left line
    Y = mnist.train.images[0:T, 14, 14:(14+n)].reshape(T, n).T  # Right line
    X += np.random.randn(m, T) * 0.05  # Add noise
    Y += np.random.randn(n, T) * 0.05  # Add noise
    X -= X.mean(axis=1, keepdims=True)  # Mean-centering
    Y -= Y.mean(axis=1, keepdims=True)  # Mean-centering
    return X, Y


def mediamill_data_from_raw(m=101, n=120, T=30993):
    """Extract Mediamill data from original files and save in numpy format.

    :param m: Dimensionality of textual feature, maximum 101.
    :param n: Dimensionality of visual feature, maximum 120.
    :param T: Number of frames, maximum 30993.
    :return: textual and visual features.
    """
    dir = "./mediamill/mediamill-challenge/Challenge/train/experiment1/features/"
    files = os.listdir(dir)

    textual = np.zeros([len(files), T])
    for i in range(len(files)):
        data = load_svmlight_file(dir + files[i])
        textual[i, :] = (data[1][0:T] + 1) // 2
    count = textual.sum(axis=1)
    ind = count.argsort()[::-1]
    for i in range(len(files)):
        print(files[ind[i]], count[ind[i]])
    textual = textual[ind[0:m]]
    assert textual.shape == (m, T), textual.shape

    visual = load_svmlight_file(dir + files[0])[0].toarray()
    visual = visual[0:T, 0:n].T

    np.savez(open("./mediamill/textual_visual.npz", mode="xb"),
             textual=textual, visual=visual)
    return textual, visual


def mediamill_data(m, n, T):
    """Extract Mediamill data from numpy(.npz) files.

    :param m: Dimensionality of textual feature, maximum 101.
    :param n: Dimensionality of visual feature, maximum 120.
    :param T: Number of frames, maximum 30993.
    :return: textual and visual features.
    """
    filename = "./mediamill/textual_visual.npz"
    data = np.load(filename)
    textual = data["textual"][:m, :T]
    visual = data["visual"][:n, :T]

    textual -= textual.mean(axis=1, keepdims=True)
    visual -= visual.mean(axis=1, keepdims=True)
    return textual, visual
