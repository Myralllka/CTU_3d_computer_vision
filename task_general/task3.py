import toolbox
from toolbox import *

THETA = 1
NUM_OF_IMGS = 12

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
            print()

    a1 = 6
    a2 = 10
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
    X = p2e(Pu2X(P1_c, P2_c, u1_correct, u2_correct))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    scale = 4
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)

    origin = np.eye(3)
    d = np.array([0, 0, 0])

    plot_csystem(ax, origin, d, 'c1')
    plot_csystem(ax, R.T, R.T @ -t, 'c2')

    ax.plot3D(X[0], X[1], X[2], 'b.')
    ax.plot3D(0, 0, 0, "r.")
    plt.show()

    # g = ge.GePly('out.ply')
    # g.points(X)  # Xall contains euclidean points (3xn matrix), ColorAll RGB colors (3xn or 3x1, optional)
    # g.points(np.array([0, 0, 0]).reshape(3, 1), color=np.array([255.0, .0, .0]).reshape(3, 1))
    # g.points((R.T @ -t).reshape(3, 1), color=np.array([.0, 255.0, .0]).reshape(3, 1))
    # g.close()
