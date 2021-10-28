import numpy as np

from toolbox import *
from mpl_toolkits import mplot3d


def Rx(theta):
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])


def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])


def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])


if __name__ == "__main__":
    # R = Rx*Ry*X
    X1 = np.array([[-0.5, 0.5, 0.5, -0.5, -0.5, -0.3, -0.3, -0.2, -0.2, 0, 0.5],
                   [-0.5, -0.5, 0.5, 0.5, -0.5, -0.7, -0.9, -0.9, -0.8, -1, -0.5],
                   [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]])

    X2 = np.array([[-0.5, 0.5, 0.5, -0.5, -0.5, -0.3, -0.3, -0.2, -0.2, 0, 0.5],
                   [-0.5, -0.5, 0.5, 0.5, -0.5, -0.7, -0.9, -0.9, -0.8, -1, -0.5],
                   [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]])

    K = np.array([[1000, 0, 500],
                  [0, 1000, 500],
                  [0, 0, 1]])

    # P = [KR | -KRC]
    # R1 == R2 == R3 == I
    C1 = np.array([0, 0, 0])
    P1 = np.c_[K, C1]

    C2 = np.array([0, -1, 0])
    P2 = np.c_[K, (-K) @ C2]

    C3 = np.array([0, 0.5, 0])
    P3 = np.c_[K, (-K) @ C3]

    C4 = np.array([0, -3, 0.5])
    R4 = Rx(0.5)
    P4 = np.c_[K @ R4, (-K) @ R4 @ C4]

    C5 = np.array([0, -5, 4.2])
    R5 = Rx(np.pi / 2)
    P5 = np.c_[K @ R5, (-K) @ R5 @ C5]

    C6 = np.array([-1.5, -3, 1.5])
    R6 = Rx(0.8) @ Ry(-0.5)
    P6 = np.c_[K @ R6, (-K) @ R6 @ C6]

    Ps = [P1, P2, P3, P4, P5, P6]

    x1 = e2p(X1.T)
    x2 = e2p(X2.T)

    counter = 0

    for p in Ps:
        fig = plt.figure(counter)
        counter += 1
        u1, v1, u2, v2 = [], [], [], []
        for i in range(len(x1)):
            plt.gca().invert_yaxis()
            plt.axis('equal')  # this kind of plots should be isotropic
            # u1, v1 = p2e([p @ x1[i]])[0]
            # u2, v2 = p2e([p @ x2[i]])[0]
            a1, b1 = p2e([p @ x1[i]])[0]
            a2, b2 = p2e([p @ x2[i]])[0]
            u1.append(a1)
            v1.append(b1)
            u2.append(a2)
            v2.append(b2)

        plt.plot(u1, v1, 'r-', linewidth=2)
        plt.plot(u2, v2, 'b-', linewidth=2)
        plt.plot([u1, u2], [v1, v2], 'k-', linewidth=2)

    # ax = plt.axes(projection='3d')
    # ax.plot3D(X1[0], X1[1], X1[2], "r-")
    # ax.plot3D(X2[0], X2[1], X2[2], "b-")
    # for i in range(len(X1[0])):
    #     ax.plot3D([X1[0][i], X2[0][i]], [X1[1][i], X2[1][i]], [X1[2][i], X2[2][i]], color="black")
    plt.show()
