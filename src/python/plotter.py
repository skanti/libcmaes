import numpy as np
import glob
import os.path
import matplotlib.pyplot as plt

dir = "/Users/amon/grive/uni/sofc/700C"
all_files = glob.glob(dir + "/700C_fitted/*.sol")
for filename in all_files:
    basename = os.path.basename(filename)
    rawname = os.path.splitext(basename)[0]
    f = open(filename, 'r')
    # -> read params
    n_params = int(f.readline())
    params = np.empty(n_params)
    for i in range(n_params):
        params[i] = float(f.readline())
    # <-

    # -> read x values
    n_data = int(f.readline())
    x = np.empty(n_data)
    for i in range(n_data):
        x[i] = float(f.readline())
    # <-

    # -> read y model values
    dim = int(f.readline())
    y_model = np.empty((n_data, dim))
    for i in range(n_data):
        y_model[i, :] = np.array([float(y) for y in f.readline().split(' ')])
    # <-

    # -> read y data values
    y_data = np.empty((n_data, dim))
    for i in range(n_data):
        y_data[i, :] = np.array([float(y) for y in f.readline().split(' ')])
    # <-

    plt.clf()
    plt.plot(y_model[:, 0], y_model[:, 1], color='b', lw=2, alpha=0.8)
    plt.scatter(y_data[:, 0], y_data[:, 1], color='r', marker='o', alpha=0.5)
    plt.ylabel("y")
    plt.xlabel("x")
    plt.gca().invert_yaxis()
    plt.savefig(dir + "/700C_fitted/" + rawname + ".png")
    # plt.legend(["y (data)"])
    # plt.show()
