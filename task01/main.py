import numpy as np  # for matrix computation and linear algebra
import matplotlib.pyplot as plt  # for drawing and image I/O
import matplotlib.patches as patches
import matplotlib.pyplot as mp
import scipy.linalg as slinalg
from toolbox import *


def plt_line(p1, p2, marker="b-"):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], marker)


def tellme(s):
    print(s)
    plt.title(s, fontsize=16)
    plt.draw()


def plt_rectangle(p1, p2, p3, p4, marker='b-'):
    plt_line(p1, p2, marker)
    plt_line(p2, p3, marker)
    plt_line(p3, p4, marker)
    plt_line(p4, p1, marker)


def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return float('inf'), float('inf')
    return x / z, y / z


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

    intersection = get_intersect(pts[0][0], pts[0][1], pts[1][0], pts[1][1])

    plt.plot(intersection[0], intersection[1], "ro", mfc='none')

    plt_line(pts[0][0], pts[0][1])
    plt_line(pts[1][0], pts[1][1], "r-")
    tellme('Happy? Key click for yes, mouse click for no')
    # numpy.cross()
    plt.show()
