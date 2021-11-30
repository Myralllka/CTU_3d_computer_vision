import numpy as np

import toolbox
from copy import deepcopy
from toolbox import *

THETA = 1
NUM_OF_IMGS = 12


class Camera:
    def __init__(self, n: int, img):
        self.n = n
        self.img = img
        self.interest_points = None

        self.has_P = False
        self.P = None
        self.R = None
        self.t = None

    def __hash__(self):
        return hash(self.img)

    def set_P(self, K, R, t):
        self.P = K @ np.c_[R, t]
        self.R = R
        self.t = t.reshape(3, )
        self.has_P = True


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
    imgs_order = [7, 11, 3, 6, 10, 2, 5, 9, 1, 4, 8, 0]

    cameras = dict()
    Es = []
    # imgs_points_arr = []

    # init K
    K = np.loadtxt('task_general/K.txt')
    K_inv = np.linalg.inv(K)

    # init images and cameras
    for i in range(NUM_OF_IMGS):
        img = plt.imread('task_general/imgs/{:02}.jpg'.format(i + 1))
        cameras[i] = Camera(i, deepcopy(img))

    c = Corresp(NUM_OF_IMGS)

    # for debug info
    c.verbose = 2

    # construct images correspondences
    for view_1 in range(NUM_OF_IMGS):
        cameras[view_1].interest_points = np.loadtxt('task_general/data/u_{:02}.txt'.format(view_1 + 1)).T
        for view_2 in range(view_1 + 1, NUM_OF_IMGS):
            points_relations = np.loadtxt('task_general/data/m_{:02}_{:02}.txt'.format(view_1 + 1,
                                                                                       view_2 + 1), dtype=int)
            c.add_pair(view_1, view_2, points_relations)

    a1 = imgs_order[0]
    a2 = imgs_order[1]

    u1 = cameras[a1].interest_points
    u2 = cameras[a2].interest_points

    u1p_K = e2p(u1)
    u2p_K = e2p(u2)
    corresp = np.array(c.get_m(a1, a2))

    #  initial pair reconstruction

    E, R, t, inliers_idxs, inliers_corresp_idxs = u2ERt_optimal(u1p_K, u2p_K, corresp, K)

    Es.append(E)

    F = K_inv.T @ (sqc(-t) @ R) @ K_inv

    # draw_epipolar_lines(u1p_K[:, inliers_idxs[0]], u2p_K[:, inliers_idxs[1]], F, imgs[a1], imgs[a2])

    # P1_c = np.c_[(K @ np.eye(3), np.zeros((3, 1)))]
    # P2_c = np.c_[(K @ R, (K @ t).reshape(3, 1))]
    cameras[a1].set_P(K, np.eye(3), np.zeros((3, 1)))
    cameras[a2].set_P(K, R, t)
    # Ps.append(P1_c)
    # Ps.append(P2_c)
    X = p2e(Pu2X_optimised(cameras[a1].P, cameras[a2].P, u1p_K, u2p_K, F, inliers_idxs))
    # init 3d points
    c.start(imgs_order[0], imgs_order[1], inliers_corresp_idxs)

    #  add one more camera

    # add all other cameras

    for i in imgs_order[2:]:
        corresp_X2u = np.array(c.get_Xu(i))[:-1]
        R2, t2, inlier_corresp_idxs = ransac_Rt_p3p(X, e2p(cameras[i].interest_points), corresp_X2u, K)
        c.join_camera(i, inlier_corresp_idxs)
        c_P = K @ np.c_[R2, t2]
        cameras[i].set_P(K, R2, t2)
        break
        # ilist = c.get_cneighbours(i)
        # for ic in ilist:
        #     mi, mic = c.get_m(i, ic)
        # new_Xs = p2e(Pu2X_optimised(P1_c, P2_c, u1p_K, u2p_K, F, inliers_idxs))

    #  make a 3d plot

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    scale = 6
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)

    origin = np.eye(3)
    d = np.array([0, 0, 0])

    # plot_csystem(ax, origin, d, 'c{}'.format(imgs_order[0]))
    for i in range(3):
        R = cameras[imgs_order[i]].R
        t = cameras[imgs_order[i]].t
        plot_csystem(ax, R.T, R.T @ -t, 'c{}'.format(imgs_order[i]))

    # R = cameras[imgs_order[1]].R
    # R2 = cameras[imgs_order[2]].R
    # t = cameras[imgs_order[1]].t
    # t2 = cameras[imgs_order[2]].t
    #
    # plot_csystem(ax, R.T, R.T @ -t, 'c{}'.format(imgs_order[1]))
    # plot_csystem(ax, R2.T, R2.T @ -t2.reshape(3, ), 'c{}'.format(imgs_order[2]))

    ax.plot3D(X[0], X[1], X[2], 'b.')
    ax.plot3D(0, 0, 0, "r.")
    plt.show()

    #  make a point cloud

    # g = ge.GePly('out.ply')
    # g.points(X)  # Xall contains euclidean points (3xn matrix), ColorAll RGB colors (3xn or 3x1, optional)
    # g.points(np.array([0, 0, 0]).reshape(3, 1), color=np.array([255.0, .0, .0]).reshape(3, 1))
    # g.points((R.T @ -t).reshape(3, 1), color=np.array([.0, 255.0, .0]).reshape(3, 1))
    # g.close()
