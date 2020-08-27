import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from CCAPyramidal import CCAPyramidal
from CCADendrites import CCADendritesOnOff
from CCADendrites import show_whiteboard_training
from CCACompare import CCAComparer
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/", reshape=False)


def illustrate_mnist():
    arr = mnist.train.images[0, :, :, 0] * 255
    arr = arr.astype(np.uint8)
    im = Image.fromarray(arr)
    assert im.mode == "L", im.mode
    im = im.convert(mode="RGB")
    pixel = im.load()
    for i in range(9, 14):
        pixel[i, 14] = (255, 0, 0)
    for i in range(14, 19):
        pixel[i, 14] = (0, 0, 255)
    plt.subplot(111)
    plt.imshow(im)
    plt.show()


def figure_compares():
    dataset = ["gaussian", "mnist", "mediamill"]
    color = [np.array([0, 0, 1]), np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([0, 1, 1])]
    labels = np.array([["a", "b"], ["c", "d"], ["e", "f"]])
    fig, axs = plt.subplots(3, 2, figsize=(9, 12))
    for i in range(3):
        comparer = pickle.load(open("./result_data/compare_{}.pickle"
                               .format(dataset[i]), mode="rb"))
        print(dataset[i])
        comparer.show_all(color, ax1=axs[i, 0], ax2=axs[i, 1], label_a=labels[i, 0], label_b=labels[i, 1], sigma=1000)
    plt.show()


def illustrate_mnist4X(ax):
    arr = mnist.train.images[0, :, :, 0] * 255
    arr = arr.astype(np.uint8)
    arr4 = np.zeros([32, 32], dtype=np.uint8)
    arr4[1:15, 1:15] = arr[0:14, 0:14]
    arr4[1:15, 17:31] = arr[0:14, 14:28]
    arr4[17:31, 1:15] = arr[14:28, 0:14]
    arr4[17:31, 17:31] = arr[14:28, 14:28]
    im = Image.fromarray(arr4)
    assert im.mode == "L", im.mode
    im = im.convert(mode="RGB")
    pixel = im.load()
    for i in range(32):
        for j in range(32):
            if 0 <= i <= 15 and 0 <= j <= 15 and not (1 <= i <= 14 and 1 <= j <= 14): pixel[i, j] = (255, 0, 0)
            if 0 <= i <= 15 and 16 <= j <= 31 and not (1 <= i <= 14 and 17 <= j <= 30): pixel[i, j] = (0, 0, 255)
            if 16 <= i <= 31 and 0 <= j <= 15 and not (17 <= i <= 30 and 1 <= j <= 14): pixel[i, j] = (0, 255, 0)
            if 16 <= i <= 31 and 16 <= j <= 31 and not (17 <= i <= 30 and 17 <= j <= 30): pixel[i, j] = (255, 255, 0)
    ax.imshow(im)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.text(-0.1, 1, "a",
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontweight="bold")


def figure_multiview():
    dendrites_onoff_mnist4X = pickle.load(open("./result_data/dendrites_onoff_mnist4X.pickle", mode="rb"))
    assert isinstance(dendrites_onoff_mnist4X, CCADendritesOnOff)

    ax = plt.subplot(3, 2, 1)
    illustrate_mnist4X(ax)

    ax = plt.subplot(6, 13, (8, 25))
    n_sqrt = int(np.sqrt(dendrites_onoff_mnist4X.w[0].shape[0]))
    T = dendrites_onoff_mnist4X.X[0].shape[1]
    ind = [(8, 9), (11, 12), (21, 22), (24, 25)]
    for i in range(4):
        plt.subplot(6, 13, ind[i])
        variances = np.zeros([n_sqrt * n_sqrt])
        for j in range(n_sqrt * n_sqrt):
            variances[j] = dendrites_onoff_mnist4X.w[i][j] \
                           * dendrites_onoff_mnist4X.X[i][[j]] @ dendrites_onoff_mnist4X.X[i][[j]].T \
                           * dendrites_onoff_mnist4X.w[i][j] / T
        plt.imshow(variances.reshape(n_sqrt, n_sqrt))
        plt.colorbar()
        plt.xticks(())
        plt.yticks(())
    plt.text(-0.05, 1, "b",
             horizontalalignment="center", verticalalignment="center", transform=ax.transAxes,
             fontweight="bold")

    ax = plt.subplot(3, 2, 3)
    ax.plot(dendrites_onoff_mnist4X.online_obj, label="online")
    ax.plot(dendrites_onoff_mnist4X.offline_obj, label="offline")
    ax.set_xlim([0, dendrites_onoff_mnist4X.steps])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Absolute objective")
    ax.text(-0.12, 1, "c",
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontweight="bold")
    ax.legend()

    ax = plt.subplot(3, 2, 4)
    ax.plot(dendrites_onoff_mnist4X.err)
    ax.set_xlim([0, dendrites_onoff_mnist4X.steps])
    ax.set_ylim([0, 90])
    ax.set_ylabel("Approximate angular error")
    ax.text(-0.1, 1, "d",
            horizontalalignment="center", verticalalignment="center", transform=ax.transAxes, fontweight="bold")

    show_whiteboard_training(ax1=plt.subplot(3, 2, 5), ax2=plt.subplot(3, 2, 6))

    plt.show()


if __name__ == "__main__":
    illustrate_mnist()  # Figure 4
    figure_compares()  # Figure 5
    figure_multiview()  # Figure 6
