import numpy as np
import random

import toolbox
from toolbox import *

ITERATIONS = 100
EPSILON = 5
random.seed(1)


def fit_line_ransac(Xs, Ys, iterations=ITERATIONS, epsilon=EPSILON):
    best_score = 0
    best_points = []
    pts = e2p(np.array([Xs, Ys]).T)
    for index in range(iterations):
        p1, p2 = random.sample(range(iterations), 2)
        p1, p2 = pts[p1], pts[p2]
        l = np.cross(p1, p2)
        l_norm = np.array([l / np.sqrt(l[0] ** 2 + l[1] ** 2)])
        distances = l_norm @ pts.T
        bools = abs(distances[0]) < epsilon
        distances_trashold = distances[0][bools]
        score = sum(distances_trashold)
        if best_score < score:
            best_score = score
            best_points = [p1, p2]
    return best_points


def ransac_lsqr(Pts, Distances):
    score = 0
    line = [1, 2]

    return score, line


def mlesac_lsqr(Pts, Distances):
    score = 0
    line = [1, 2]

    return score, line


def fit_line_sac(Xs, Ys, vote_func, iterations=ITERATIONS, epsilon=EPSILON):
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
            best_points = line
        # if best_support < sum(distances_threshold):
        #     best_support = sum(distances_threshold)
        #     k, b = np.polyfit(pts_treshold.T[0], pts_treshold.T[1], deg=1)
        #     best_points_kb = [p1, p2, k, b]
    return best_points[0], best_points[1]


#
#
# def fit_line_mlesac_lsqr(Xs, Ys, iterations=ITERATIONS, epsilon=EPSILON):
#     best_support = 0
#     best_points_kb = []
#     pts = e2p(np.array([Xs, Ys]).T)
#     for index in range(iterations):
#         p1, p2 = random.sample(range(iterations), 2)
#         p1, p2 = pts[p1], pts[p2]
#         l = np.cross(p1, p2)
#         l_norm = np.array([l / np.sqrt(l[0] ** 2 + l[1] ** 2)])
#         distances = l_norm @ pts.T
#         bools = abs(distances[0]) < epsilon
#         distances_threshold = 1 - (distances[0][bools] ** 2) / epsilon ** 2
#         pts_treshold = pts[bools]
#         if best_support < sum(distances_threshold[0]):
#             best_support = distances_threshold.shape[0]
#             k, b = np.polyfit(pts_treshold.T[0], pts_treshold.T[1], deg=1)
#             best_points_kb = [p1, p2, k, b]
#
#     return best_points_kb[2], best_points_kb[3]


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
    p1, p2 = fit_line_ransac(points[0], points[1])
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
    #
    # # Plot mlesac + lst sqr
    # k, b = fit_line_ransac_lsqr(points[0], points[1])
    # line = [k, -1, b]
    # y = np.array([0, 300])
    # x = (-line[2] / line[0]) + (-line[1] / line[0]) * y
    # plt.plot(x, y, 'c-', label="mlesac + lstqt")

    plt.legend()
    plt.axis([-50, 500, -100, 400])
    plt.show()
