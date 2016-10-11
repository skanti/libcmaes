import numpy as np
import glob
import os.path
import matplotlib.pyplot as plt
import scipy.signal as spsig


class EISData:
    n = None  # <- number of sampled frequencies
    w = None  # <- frequencies
    z = None  # <- impedances

    @staticmethod
    def plot_data():
        plt.scatter(EISData.z.real, EISData.z.imag)
        plt.ylabel("Z''")
        plt.xlabel("Z'")
        plt.legend(["z (data)"])
        plt.gca().invert_yaxis()
        plt.draw()
        plt.show()

    @staticmethod
    def load_data(filename):
        f = open(filename, 'r')
        a = np.loadtxt(f, skiprows=125)
        va = a[0, 1]
        real = a[:, 4]
        imag = a[:, 5]
        freq = a[:, 0]
        # -> filter
        args = np.where(freq < 1e5)[0]
        EISData.w = a[args, 0]
        EISData.n = np.size(EISData.w, axis=0)
        EISData.z = np.empty(EISData.n, dtype=np.complex)
        EISData.z.imag = imag[args]
        EISData.z.real = real[args]
        # <-

    @staticmethod
    def filter(n_kernel):
        s = int(n_kernel / 2)  # <- margin
        e = s + 1
        # -> copy
        z = np.copy(EISData.z)
        # <-
        # -> filter
        z.real = spsig.medfilt(z.real, n_kernel + 2)
        z.imag = spsig.medfilt(z.imag, n_kernel + 2)
        z[:e] = EISData.z[:e]
        z[-e:] = EISData.z[-e:]
        z.real = spsig.wiener(z.real, mysize=n_kernel, noise=0.01)
        z.imag = spsig.wiener(z.imag, mysize=n_kernel, noise=0.01)

        # <-
        # -> crop
        EISData.z[s:-s] = z[s:-s]
        # <-


dir = "/Users/amon/grive/uni/sofc/denso"
all_files = glob.glob(dir + "/denso_raw/*.z")
i = 0
for filename in all_files:
    basename = os.path.basename(filename)
    rawname = os.path.splitext(basename)[0]
    filename_i = dir + '/denso_filtered/' + rawname + '.dat'
    # if os.path.isfile(filename_i) is not True:
    print(rawname)
    EISData.load_data(filename)
    plt.figure(figsize=(15, 5))
    #plt.subplot(1, 2, 1)
    plt.gca().invert_yaxis()
    plt.scatter(EISData.z.real, EISData.z.imag, color='r', marker='o', alpha=0.5)
    EISData.filter(3)
    #plt.subplot(1, 2, 2)
    plt.scatter(EISData.z.real, EISData.z.imag, color='g', marker='o', alpha=0.5)
    #plt.gca().invert_yaxis()
    #plt.tight_layout()
    plt.show()
    #np.savetxt(filename_i, [EISData.n], fmt='%d')
    #f = open(filename_i, 'ab')
    #np.savetxt(f, EISData.w)
    #np.savetxt(f, EISData.z.real)
    #np.savetxt(f, EISData.z.imag)
    #f.close()
