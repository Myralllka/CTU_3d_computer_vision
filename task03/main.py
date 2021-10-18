import numpy as np
import random

import toolbox
from toolbox import *

ITERATIONS = 100
EPSILON = 5
random.seed(1)


def ransac(Pts, Distances):
    return Pts.shape[0], None


def ransac_lsqr(Pts, Distances):
    score = Pts.shape[0]
    return score, np.polyfit(Pts.T[0], Pts.T[1], deg=1)


def mlesac_lsqr(Pts, Distances):
    score = sum(1 - (Distances ** 2) / EPSILON ** 2)
    return score, np.polyfit(Pts.T[0], Pts.T[1], deg=1)


def fit_line_sac(Xs, Ys, vote_func, iterations=ITERATIONS, epsilon=EPSILON):
    random.seed(2)
    best_support = 0
    best_points = []
    pts = e2p(np.array([Xs, Ys]).T)
    for index in range(iterations):
        p1, p2 = random.sample(range(iterations), 2)
        p1, p2 = pts[p1], pts[p2]
        l = np.cross(p1, p2)
        l_norm = np.array([l / np.sqrt(l[0] ** 2 + l[1] ** 2)])
        distances = l_norm @ pts.T
        bools = abs(distances[0]) < epsilon
        distances_threshold = distances[0][bools]
        pts_treshold = pts[bools]

        score, line = vote_func(pts_treshold, distances_threshold)
        if best_support < score:
            best_support = score
            if line is None:
                best_points = [p1, p2]
            else:
                best_points = line
    return best_points[0], best_points[1]


if __name__ == "__main__":
    # plt.axis('equal')
    # Plot an original line
    line = np.array([-10, 3, 1200])
    y = np.array([0, 300])
    x = (-line[2] / line[0]) + (-line[1] / line[0]) * y
    plt.plot(x, y, 'k-', label="original")

    # Plot input points with noise
    points = np.loadtxt('task03/data/linefit_3.txt').T
    plt.plot(points[0], points[1], '.')

    # Fit the line using non-robust approach
    k, b = np.polyfit(points[0], points[1], deg=1)
    line = [k, -1, b]
    y = np.array([0, 300])
    x = (-line[2] / line[0]) + (-line[1] / line[0]) * y
    plt.plot(x, y, 'y-', label="lstqt")

    # plot ransac fitting line
    p1, p2 = fit_line_sac(points[0], points[1], ransac)
    line = np.cross(p1, p2)
    y = np.array([0, 300])
    x = (-line[2] / line[0]) + (-line[1] / line[0]) * y
    plt.plot(x, y, 'r-', label="ransac")

    # Plot ransac + lst sqr
    k, b = fit_line_sac(points[0], points[1], ransac_lsqr)
    line = [k, -1, b]
    y = np.array([0, 300])
    x = (-line[2] / line[0]) + (-line[1] / line[0]) * y
    plt.plot(x, y, 'g-', label="ransac + lstqt")

    # Plot mlesac + lst sqr
    k, b = fit_line_sac(points[0], points[1], mlesac_lsqr)
    line = [k, -1, b]
    y = np.array([0, 300])
    x = (-line[2] / line[0]) + (-line[1] / line[0]) * y
    plt.plot(x, y, 'c-', label="mlesac + lstqt")

    plt.legend()
    plt.axis([-50, 500, -100, 400])
    plt.show()
