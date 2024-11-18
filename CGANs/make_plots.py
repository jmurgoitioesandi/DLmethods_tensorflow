import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def save_plot_histogram2d(fake, true_X, true_Y, savefolder, epoch):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    rangeHist2d = np.array([[-3, 3], [-2, 2]])
    ax[0].hist2d(fake[:, 0], fake[:, 1], bins=100, range=rangeHist2d)
    ax[1].hist2d(true_X[:, 0], true_Y[:, 0], bins=100, range=rangeHist2d)
    ax[1].set_ylim(-2, 2.0)
    ax[0].set_ylim(-2, 2.0)
    ax[0].set_xlabel("x", fontsize=14)
    ax[0].set_ylabel("y", fontsize=14)
    ax[1].set_xlabel("x", fontsize=14)
    ax[1].set_ylabel("y", fontsize=14)
    ax[0].set_title("Fake", fontsize=18)
    ax[1].set_title("True", fontsize=18)
    plt.savefig(savefolder + "/Histogram2D_epochs=" + str(epoch))
    plt.close()
