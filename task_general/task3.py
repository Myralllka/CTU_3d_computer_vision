import numpy as np

import toolbox
from toolbox import *

THETA = 2
NUM_OF_IMGS = 12


def plot_csystem(ax, base, origin, name):
    """
    drawing a coordinate system with base Base located in the origin b with a
    given name and color. The base and origin are expressed in the world
    coordinate system δ. The base consists of a two or three three-dimensional
    column vectors of coordinates. E.g.
    hw03.plot_csystem(ax,np.eye(3),np.zeros([3,1]),'k','d')
    δ_x, δ_y, δ_z
    :param base: 3x2 or 3x3 mat
    :param origin: 1x3 vec
    """
    ax.quiver3D(origin[0, 0], origin[1, 0], origin[2, 0],
                base[0][0],
                base[1][0],
                base[2][0],
                arrow_length_ratio=0.1,
                length=1,
                color="red")
    ax.text(base[0][0] + origin[0, 0],
            base[1][0] + origin[1, 0],
            base[2][0] + origin[2, 0],
            name + "_x")
    ax.quiver3D(origin[0], origin[1], origin[2],
                base[0][1] - origin[0, 0],
                base[1][1] - origin[1, 0],
                base[2][1] - origin[2, 0],
                arrow_length_ratio=0.1,
                length=1,
                color="green")
    ax.text(base[0][1], base[1][1], base[2][1], name + "_y")
    if base.shape[1] > 2:
        ax.quiver3D(origin[0, 0], origin[1, 0], origin[2, 0],
                    base[0][2] - origin[0, 0],
                    base[1][2] - origin[1, 0],
                    base[2][2] - origin[2, 0],
                    arrow_length_ratio=0.1,
                    length=1,
                    color="blue")
        ax.text(base[0][2], base[1][2], base[2][2], name + "_z")


if __name__ == "__main__":
    ### Preparing, loading the data
    ### cameras located like this:
    ### 1--2--3--4
    ### |  |  |  |
    ### 5--6--7--8
    ### |  |  |  |
    ### 9-10-11-12
    ### numeration from 0, so in the code they will be:
    ### 0--1--2--3
    ### |  |  |  |
    ### 4--5--6--7
    ### |  |  |  |
    ### 8--9-10-11
    ### So the best choice for the very beginning is the pair 6--7
    ### Data setup
    imgs = []
    imgs_todo = set(range(NUM_OF_IMGS))
    imgs_done = set()
    Xs = []
    imgs_points_arr = []

    # init K
    K = np.loadtxt('task_general/K.txt')
    K_inv = np.linalg.inv(K)

    # init images
    for i in range(NUM_OF_IMGS):
        imgs.append(plt.imread('task_general/imgs/{:02}.jpg'.format(i + 1)))
    imgs = np.array(imgs)

    c = Corresp(NUM_OF_IMGS)

    # for debug info
    # c.verbose = 2

    # construct images correspondences
    for view_1 in range(NUM_OF_IMGS):
        imgs_points_arr.append(np.loadtxt('task_general/data/u_{:02}.txt'.format(view_1 + 1)).T)
        for view_2 in range(view_1 + 1, NUM_OF_IMGS):
            points_relations = np.loadtxt('task_general/data/m_{:02}_{:02}.txt'.format(view_1 + 1,
                                                                                       view_2 + 1), dtype=int)
            c.add_pair(view_1, view_2, points_relations)

    a1 = 5
    a2 = 6
    imgs_todo.remove(a1)
    imgs_todo.remove(a2)

    u1 = imgs_points_arr[a1]
    u2 = imgs_points_arr[a2]

    u1p_K = e2p(u1)
    u2p_K = e2p(u2)
    corresp = np.array(c.get_m(a1, a2))
    E, R, t, inliers_idxs, inliers_corresp_idxs = u2ERt_optimal(u1p_K, u2p_K, corresp, K)

    F = K_inv.T @ (sqc(-t) @ R) @ K_inv

    draw_epipolar_lines(u1p_K[:, inliers_idxs[0]], u2p_K[:, inliers_idxs[1]], F, imgs[a1], imgs[a2])

    u1_correct, u2_correct = u_correct_sampson(F, u1p_K[:, inliers_idxs[0]], u2p_K[:, inliers_idxs[1]])
    # u1_correct, u2_correct = u1p_K[:, inliers_idxs[0]], u2p_K[:, inliers_idxs[1]]

    P1_c = np.c_[(K @ np.eye(3), np.zeros((3, 1)))]
    P2_c = np.c_[(K @ R, (K @ t).reshape(3, 1))]
    X = Pu2X(P1_c, P2_c, u1_correct, u2_correct)
    X = p2e(X)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)

    Delta = np.eye(3)

    d = np.array([0, 0, 0]).reshape(3, 1)

    plot_csystem(ax, Delta @ np.linalg.inv(R), d, 'δ')

    # d = np.array([[d[0]], [d[1]], [d[2]]])

    ax.plot3D(X[0], X[1], X[2], 'b.')
    ax.plot3D(0, 0, 0, "r.")
    # origin = K @ np.eye(3)
    # C = np.array([0, 0, 0]).reshape(3, 1)
    # plot_csystem(ax, origin, np.array([0, 0, 0]).reshape(3, 1), '1', 'red')

    # origin = K @ R
    # C =
    # plot_csystem(ax, origin, (-K @ R @ C).reshape(3, 1), '2', 'green')

    plt.show()

    g = ge.GePly('out.ply')
    g.points(X)  # Xall contains euclidean points (3xn matrix), ColorAll RGB colors (3xn or 3x1, optional)
    #
    # # g.points(u1p_K, color=np.array([100.0, 255.0, 100.0]).reshape(3, 1))
    # # g.points(u2p_K, color=np.array([100.0, 255.0, 100.0]).reshape(3, 1))
    #
    g.points(np.array([0, 0, 0]).reshape(3, 1), color=np.array([255.0, .0, .0]).reshape(3, 1))
    g.points(t.reshape(3, 1), color=np.array([.0, 255.0, .0]).reshape(3, 1))
    g.close()
