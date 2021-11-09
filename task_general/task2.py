import numpy as np

import toolbox
from toolbox import *

THETA = 0.0001

colors = ["dimgray", "rosybrown", "maroon", "peru",
          "moccasin", "yellow", "olivedrab", "lightgreen",
          "navy", "royalblue", "indigo", "hotpink"]


def draw_epipolar_lines(u1_step2, u2_step2, F_step2, header='The epipolar lines using F'):
    fig = plt.figure()
    fig.clf()
    fig.suptitle(header)
    plt.subplot(121)

    i = 0
    # MAX = 50
    # n = 50
    for x_p1, y_p1, x_p2, y_p2 in zip(u1_step2[0], u1_step2[1], u2_step2[0],
                                      u2_step2[1]):
        # if n != MAX:
        #     n += 1
        #     continue
        # n = 0
        plt.plot([int(x_p1)], [int(y_p1)], color=colors[i], marker="X",
                 markersize=10)
        point2_step2 = np.c_[x_p2, y_p2, 1].reshape(3, 1)

        x = np.linspace(1, img1.shape[1], img1.shape[1])
        ep1_step2 = F_step2.T @ point2_step2
        y = -((ep1_step2[2] / ep1_step2[1]) + x * ep1_step2[0] / ep1_step2[1])
        plt.plot(x, y, color=colors[i])

        i = (i + 1) % (len(colors) - 1)
    plt.imshow(img1)

    plt.subplot(122)

    i = 0
    # n = 0
    for x_p1, y_p1, x_p2, y_p2 in zip(u1_step2[0], u1_step2[1], u2_step2[0], u2_step2[1]):
        # if n != MAX:
        #     n += 1
        #     continue
        # n = 0
        plt.plot([int(x_p2)], [int(y_p2)],
                 color=colors[i],
                 marker="X",
                 markersize=10)
        point1_step2 = np.c_[x_p1, y_p1, 1].reshape(3, 1)

        x = np.linspace(1, img1.shape[1], img1.shape[1])
        point1_step2 = point1_step2.reshape(3, 1)
        ep2_step2 = F_step2 @ point1_step2
        y = -((ep2_step2[2] / ep2_step2[1]) + x * ep2_step2[0] / ep2_step2[1])
        plt.plot(x, y, color=colors[i])
        i = (i + 1) % (len(colors) - 1)
    plt.imshow(img2)
    plt.show()


def ransac_E(c_u1p_K, c_u2p_K, iterations=1000):
    best_score = 0
    best_R = []
    best_C = []
    best_E = []
    best_idxs = []

    c_u1p_K_undone = K_inv @ c_u1p_K
    c_u1p_K_undone /= c_u1p_K_undone[-1]
    c_u2p_K_undone = K_inv @ c_u2p_K
    c_u2p_K_undone /= c_u2p_K_undone[-1]

    for i in range(iterations):
        idxs = random.sample(range(c_u2p_K.shape[1]), 5)
        loop_u1p = c_u1p_K_undone[:, idxs]
        loop_u2p = c_u2p_K_undone[:, idxs]
        Es = p5.p5gb(loop_u1p, loop_u2p)

        for E in Es:
            R_c, t_c = Eu2Rt(E, loop_u1p, loop_u1p)
            F = K_inv.T @ E @ K_inv

            e = err_epipolar(F, c_u1p_K, c_u2p_K)
            e = e < THETA

            # TODO: compute inlines in front of camera or back

            # TODO: use scipy.optimize.fmin
            # TODO: use rodrigues rotation formula

            if np.count_nonzero(e) > best_score:
                best_score = np.count_nonzero(e)
                best_C = t_c
                best_R = R_c
                best_E = E
                # best_idxs = np.where(e)
                best_idxs = np.array(idxs)
                print(best_score)

    # return best_E, best_R, best_C, np.array(best_idxs).reshape(len(best_idxs[0]), )
    return best_E, best_R, best_C, np.array(best_idxs).reshape(best_idxs.shape[0], )


if __name__ == "__main__":
    ### Preparing, loading the data
    view_1 = 7
    view_2 = 8

    points_view_1 = np.loadtxt('task_general/data/u_{:02}.txt'.format(view_1)).T
    points_view_2 = np.loadtxt('task_general/data/u_{:02}.txt'.format(view_2)).T
    points_1_2_relations = np.loadtxt('task_general/data/m_{:02}_{:02}.txt'.format(view_1, view_2), dtype=int).T
    print(points_1_2_relations.shape)

    img1 = plt.imread('task_general/imgs/{:02}.jpg'.format(view_1))
    img2 = plt.imread('task_general/imgs/{:02}.jpg'.format(view_2))

    img1 = img1.copy()
    img2 = img2.copy()

    K = np.loadtxt('task_general/K.txt')
    K_inv = np.linalg.inv(K)

    u1 = points_view_1[:, points_1_2_relations[0]]
    u2 = points_view_2[:, points_1_2_relations[1]]

    u1p_K = e2p(u1)
    u2p_K = e2p(u2)

    ### undone K for working with F
    # u1p_K_undone = K_inv @ u1p_K
    # u1p_K_undone /= u1p_K_undone[-1]
    # u2p_K_undone = K_inv @ u2p_K
    # u2p_K_undone /= u2p_K_undone[-1]

    E, R, C, idx = ransac_E(u1p_K, u2p_K, 200)

    # compute sampson error
    # optimize
    F = K_inv.T @ E @ K_inv

    draw_epipolar_lines(u1p_K[:, idx], u2p_K[:, idx], F)
