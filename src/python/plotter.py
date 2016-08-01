import numpy as np
import matplotlib.pyplot as plt

# A = np.loadtxt("/Users/amon/grive/uni/sofc/700C/700C_filtered", delimiter=" ", dtype=np.float)
f = open("/Users/amon/grive/uni/sofc/700C/700C_fitted/test.dat", 'r')
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

plt.plot(y_model[:, 0], y_model[:, 1], color='b', lw=2, alpha=0.8)
plt.scatter(y_data[:, 0], y_data[:, 1], color='r', marker='o', alpha=0.5)
plt.ylabel("y")
plt.xlabel("x")
plt.gca().invert_yaxis()
# plt.legend(["y (data)"])
plt.show()
