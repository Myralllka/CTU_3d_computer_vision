import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import matplotlib.patches as patches
import matplotlib.pyplot as mp
import scipy.linalg as slinalg
from toolbox import *


def plt_line(p1, p2, marker="b-"):
    plt.plot(p1, p2, marker)


def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()


def plt_rectangle(p1, p2, p3, p4, marker='b-'):
    plt_line((p1[0], p2[0]), (p1[1], p2[1]), marker)
    plt_line((p2[0], p3[0]), (p2[1], p3[1]), marker)
    plt_line((p3[0], p4[0]), (p3[1], p4[1]), marker)
    plt_line((p4[0], p1[0]), (p4[1], p1[1]), marker)


if __name__ == "__main__":
    fig = plt.figure()
    plt_rectangle((1, 1), (1, 600), (800, 600), (800, 1), 'k-')
    arr = mp.ginput()
    pts = []
    tellme('Select 2 lines with mouse')
    for i in range(4):
        tellme('select point {}'.format(i))
        pts.append(plt.ginput(3, timeout=-1))
    plt_line(pts[0], pts[1])
    plt_line(pts[2], pts[3], "r-")
    tellme('Happy? Key click for yes, mouse click for no')
    # numpy.cross()
    plt.show()
