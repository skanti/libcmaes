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
    def filter():
        EISData.z.real = spsig.medfilt(EISData.z.real)
        EISData.z.imag = spsig.medfilt(EISData.z.imag)


dir = "/Users/amon/grive/uni/sofc/700C"
all_files = glob.glob(dir + "/700C_raw/*.z")
for filename in all_files:
    basename = os.path.basename(filename)
    rawname = os.path.splitext(basename)[0]
    filename_i = dir + '/700C_filtered/' + rawname + '.dat'
    if os.path.isfile(filename_i) is not True:
        EISData.load_data(filename)
        EISData.filter()
        np.savetxt(filename_i, [EISData.n], fmt='%d')
        f = open(filename_i, 'ab')
        np.savetxt(f, EISData.w)
        np.savetxt(f, EISData.z.real)
        np.savetxt(f, EISData.z.imag)
        f.close()
