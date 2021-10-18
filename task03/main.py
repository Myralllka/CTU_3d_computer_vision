import numpy as np

from toolbox import *


def fit_line(Xs, Ys):
    X_ = sum(Xs)
    Y_ = sum(Ys)
    x_Xs = Xs - X_
    y_Ys = Ys - Y_
    xXyY = x_Xs * y_Ys
    x_X2 = x_Xs * x_Xs
    m_k = sum(xXyY) / sum(x_X2)
    m_b = Y_ - (m_k * X_)
    return m_k, m_b


def fit_line_ransac(Xs, Ys, iterations=100):
    rng = np.random.default_rng(10)
    for i in range(iterations):
        i = rng.choice(Xs.size(), 2, replace=False)
    pass


if __name__ == "__main__":
    line = np.array([-10, 3, 1200])
    points = np.loadtxt('task03/data/linefit_3.txt').T
    plt.plot(points[0], points[1], '.')
    plt.axis('equal')

    k, b = fit_line(points[0], points[1])
    plt.plot(points[0], k * points[0] + b, 'r.')

    # plot a line
    y = np.array([0, 300])
    x = (-line[2] / line[0]) + (-line[1] / line[0]) * y
    plt.plot(x, y, 'k-')

    plt.show()
