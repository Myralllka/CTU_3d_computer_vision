import numpy as np

from toolbox import *


def fit_line(Xs, Ys):
    X_ = sum(Xs)
    Y_ = sum(Ys)
    x_Xs = Xs - X_
    y_Ys = Ys - Y_
    xXyY = x_Xs * y_Ys
    x_X2 = x_Xs * x_Xs
    k = sum(xXyY) / sum(x_X2)
    b = Y_ - (k * X_)
    return k, b


if __name__ == "__main__":
    points = np.loadtxt('task03/data/linefit_3.txt').T
    # points = np.array([[3, 4, 2, 6], [0, 1, 2, 3]])
    plt.plot(points[0], points[1], '.')

    k, b = fit_line(points[0], points[1])
    np_k, np_b = np.polyfit(points[0], points[1], deg=1)

    plt.plot(points[0],  k * points[0] + b, 'r.')
    plt.plot(points[0],  np_k * points[0] + np_b, 'r.')

    plt.show()
