import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt("/Users/amon/grive/development/CMA-ES/build/out.dat", delimiter=" ", dtype=np.float)
x = A[:, 0]
y = A[:, 1]
print(x)
n = int(np.sqrt(np.size(x)))
plt.scatter(x, y, color='g', alpha=0.9)
plt.ylabel("y")
plt.xlabel("x")
plt.legend(["y (data)"])
plt.show()
