import numpy as np
import scipy.optimize

import toolbox
from toolbox import *

THETA = 2


def compute_epipolar_error(vector, m_u1p, m_u2p, corres):
    """
    @param vector: [0:3] rotation
    @param vector: [3:] translation
    """
    theta = THETA
    l_u1p = m_u1p[:, corres[0]]
    l_u2p = m_u2p[:, corres[1]]
    R = mrd2R(vector[0:3])
    E = R @ sqc(- R.T @ vector[3:])
    e = err_F_sampson(E, l_u1p, l_u2p)
    e = [((el ** 2) / (theta ** 2)) - 1 if el < theta else 1 for el in e]
    return np.sum(e)


if __name__ == "__main__":
    ### Preparing, loading the data
    view_1 = 8
    view_2 = 12

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

    u1 = points_view_1
    u2 = points_view_2

    u1p_K = e2p(u1)
    u2p_K = e2p(u2)
    # s - number of parameters to find (R, C)
    # eps - the fraction of inliers among outliers
    # P - probability that the last proposal is all-inlier
    s = 12
    eps = 0.6
    P = 0.99
    E, R, C, inliers_corresp_idxs = ransac_E(u1p_K, u2p_K, points_1_2_relations, K, THETA, p5.p5gb,
                                             iterations=int(np.log10(1 - P) / np.log10(1 - eps ** s)))

    inliers_idxs = points_1_2_relations[:, inliers_corresp_idxs]

    input_rotation_C = np.concatenate((R2mrp(R), C))
    res = scipy.optimize.fmin(compute_epipolar_error, input_rotation_C, (K_inv @ u1p_K, K_inv @ u2p_K, inliers_idxs),
                              xtol=10e-10)
    print(R)
    n_R = mrd2R(res[0:3])
    print(n_R)
    print(C)
    print(res[3:])

    new_E = R @ sqc(- n_R.T @ res[3:])
    # F = K_inv.T @ new_E @ K_inv
    draw_epipolar_lines(u1p_K[:, inliers_idxs[0]], u2p_K[:, inliers_idxs[1]], K_inv.T @ E @ K_inv, img1, img2)
    draw_epipolar_lines(u1p_K[:, inliers_idxs[0]], u2p_K[:, inliers_idxs[1]], K_inv.T @ new_E @ K_inv, img1, img2)

    ### draw inliers and outliers
    fig = plt.figure(3)
    fig.clf()
    fig.suptitle("inliers and outliers")

    a = u1[:, points_1_2_relations[0, ~inliers_corresp_idxs]]
    b = u2[:, points_1_2_relations[1, ~inliers_corresp_idxs]]

    plt.plot(a[0], a[1], 'k.', markersize=.8)
    plt.plot([a[0], b[0]], [a[1], b[1]], 'k-', linewidth=.2)

    a = [u1[0, inliers_idxs[0]], u1[1, inliers_idxs[0]]]
    b = [u2[0, inliers_idxs[1]], u2[1, inliers_idxs[1]]]
    plt.plot(a[0], a[1], 'c.')
    plt.plot([a[0], b[0]], [a[1], b[1]], 'c-')
    plt.imshow(img1)
    plt.show()
