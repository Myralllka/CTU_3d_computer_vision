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


def distance(a, b):
    # a[x, y], b[x, y]
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def is_between(a, c, b):
    return distance(a, c) + distance(c, b) == distance(a, b)


def is_point_on_rect(rect, point):
    # rect: lu ru rd ld
    return (is_between(rect[0], point, rect[1]) or
            is_between(rect[1], point, rect[2]) or
            is_between(rect[2], point, rect[3]) or
            is_between(rect[3], point, rect[0]))


def is_point_in_rect(rect, point):
    x1, y1 = rect[0]
    x2, y2 = rect[2]
    x, y = point
    if x1 < x < x2:
        if y1 < y < y2:
            return True
    return False


def plt_line_in_rectangle(rect, a_local, b_local, marker='b-'):
    cs, cs_new = [], []
    cs.append(get_intersect_lines(rect[0], rect[1], a_local, b_local))
    cs.append(get_intersect_lines(rect[1], rect[2], a_local, b_local))
    cs.append(get_intersect_lines(rect[2], rect[3], a_local, b_local))
    cs.append(get_intersect_lines(rect[3], rect[0], a_local, b_local))
    for each in cs:
        if is_point_on_rect(rect, each):
            cs_new.append(each)

    plt.plot([cs_new[0][0], cs_new[1][0]], [cs_new[0][1], cs_new[1][1]])


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
    if is_point_in_rect(bound_points.T, intersection):
        plt.plot(intersection[0], intersection[1], "ro", mfc='none')

    plt_line_in_rectangle(bound_points.T, pts[0][0], pts[0][1])
    plt_line_in_rectangle(bound_points.T, pts[1][0], pts[1][1])
    tellme('intersection')
    # numpy.cross()
    plt.show()
