import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import matplotlib.patches as patches
import matplotlib.pyplot as mp
import scipy.linalg as slinalg
from toolbox import get_intersect_lines


def plt_line(p1, p2, marker="b-"):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], marker)


def tellme(s):
    plt.title(s, fontsize=16)
    plt.draw()


def plt_rectangle(p1, p2, p3, p4, marker='b-'):
    plt_line(p1, p2, marker)
    plt_line(p2, p3, marker)
    plt_line(p3, p4, marker)
    plt_line(p4, p1, marker)


if __name__ == "__main__":
    fig = plt.figure()
    bound_points = np.array([[1, 1, 800, 800], [1, 600, 600, 1]])

    plt_rectangle(bound_points.T[0],
                  bound_points.T[1],
                  bound_points.T[2],
                  bound_points.T[3],
                  marker='k-')

    arr = mp.ginput()
    pts = []
    colors = ['b', 'g', 'r', 'k']
    tellme('Select 2 lines with mouse')
    for i in range(2):
        tellme('select point {}'.format(i + 1))
        a, b = plt.ginput(2, timeout=-1)  # a(x, y), b(x, y) - points
        plt.plot(a[0], a[1], '{}o'.format(colors[i]), mfc='none')
        plt.plot(b[0], b[1], '{}o'.format(colors[i]), mfc='none')
        pts.append((a, b))

    intersection = get_intersect_lines(pts[0][0], pts[0][1], pts[1][0], pts[1][1])

    plt.plot(intersection[0], intersection[1], "ro", mfc='none')

    plt_line(pts[0][0], pts[0][1], "{}-".format(colors[0]))
    plt_line(pts[1][0], pts[1][1], "{}-".format(colors[1]))
    tellme('intersection')
    # numpy.cross()
    plt.show()
