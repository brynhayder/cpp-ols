import os
import numpy as np

import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from make_data import base_path

if __name__ == "__main__":
    data = np.loadtxt(
        fname=os.path.join(base_path, "single_input.csv"),
        delimiter=','
    )

    X = data[:, 0]
    y = data[:, 1]

    predictions = np.loadtxt(
        os.path.join(base_path, "single_input.fittedvalues")
    )

    fig, ax = plt.subplots(1)
    ax.plot(X, y, marker='.', ls='', label="data")
    ax.plot(X, predictions, label="predictions")
    ax.grid(alpha=0.2)
    ax.legend()
    plt.show()
