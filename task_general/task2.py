import toolbox
from toolbox import *

THETA = 0.5

colors = ["dimgray", "rosybrown", "maroon", "peru",
          "moccasin", "yellow", "olivedrab", "lightgreen",
          "navy", "royalblue", "indigo", "hotpink"]


def draw_epipolar_lines(c_u1, c_u2, c_F, header='The epipolar lines using F'):
    idxs = random.sample(range(c_u1.shape[1]), len(colors))
    c_u1 = c_u1[:, idxs]
    c_u2 = c_u2[:, idxs]
    fig = plt.figure()
    fig.clf()
    fig.suptitle(header)
    plt.subplot(121)
    i = 0
    for x_p1, y_p1, x_p2, y_p2 in zip(c_u1[0], c_u1[1], c_u2[0], c_u2[1]):
        plt.plot([int(x_p1)], [int(y_p1)], color=colors[i], marker="X",
                 markersize=10)
        point2_step2 = np.c_[x_p2, y_p2, 1].reshape(3, 1)

        x = np.linspace(1, img1.shape[1], img1.shape[1])
        ep1_step2 = c_F.T @ point2_step2
        y = -((ep1_step2[2] / ep1_step2[1]) + x * ep1_step2[0] / ep1_step2[1])
        plt.plot(x, y, color=colors[i])

        i += 1
    plt.imshow(img1)

    plt.subplot(122)
    i = 0
    for x_p1, y_p1, x_p2, y_p2 in zip(c_u1[0], c_u1[1], c_u2[0], c_u2[1]):
        plt.plot([int(x_p2)], [int(y_p2)],
                 color=colors[i],
                 marker="X",
                 markersize=10)
        point1_step2 = np.c_[x_p1, y_p1, 1].reshape(3, 1)

        x = np.linspace(1, img1.shape[1], img1.shape[1])
        point1_step2 = point1_step2.reshape(3, 1)
        ep2_step2 = c_F @ point1_step2
        y = -((ep2_step2[2] / ep2_step2[1]) + x * ep2_step2[0] / ep2_step2[1])
        plt.plot(x, y, color=colors[i])
        i += 1

    plt.imshow(img2)
    plt.show()


def ransac_E(c_u1p_K, c_u2p_K, iterations=1000):
    best_score = 0
    best_R = []
    best_C = []
    best_E = []
    best_idxs = []
    inliers_E, outliers_E = [], []

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
            e = err_epipolar(K_inv.T @ E @ K_inv, c_u1p_K, c_u2p_K)
            e = e < THETA
            if np.count_nonzero(e) > best_score:
                R_c, t_c = Eu2Rt(E, loop_u1p, loop_u1p)
                best_score = np.count_nonzero(e)
                best_C = t_c
                best_R = R_c
                best_E = E
                inliers_E = (c_u1p_K[:, e], c_u2p_K[:, e])
                outliers_E = (c_u1p_K[:, ~e], c_u2p_K[:, ~e])
                print(best_score)

    return best_E, best_R, best_C, inliers_E, outliers_E


if __name__ == "__main__":
    ### Preparing, loading the data
    view_1 = 4
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

    u1 = points_view_1[:, points_1_2_relations[0]]
    u2 = points_view_2[:, points_1_2_relations[1]]

    u1p_K = e2p(u1)
    u2p_K = e2p(u2)

    ### undone K for working with F
    # u1p_K_undone = K_inv @ u1p_K
    # u1p_K_undone /= u1p_K_undone[-1]
    # u2p_K_undone = K_inv @ u2p_K
    # u2p_K_undone /= u2p_K_undone[-1]

    E, R, C, inliers_E, outliers_E = ransac_E(u1p_K, u2p_K, 200)

    # compute sampson error
    # optimize
    F = K_inv.T @ E @ K_inv

    # draw_epipolar_lines(u1p_K[:, idx], u2p_K[:, idx], F)
    draw_epipolar_lines(inliers_E[0], inliers_E[1], F)

    # TODO: use scipy.optimize.fmin
    # TODO: use rodrigues rotation formula

    ### draw inliers and outliers
    fig = plt.figure(3)
    fig.clf()
    fig.suptitle("inliers and outliers")

    a = p2e(outliers_E[0])
    b = p2e(outliers_E[1])
    plt.plot(a[0], a[1], 'k.', markersize=.8)
    plt.plot([a[0], b[0]], [a[1], b[1]], 'k-', linewidth=.2)

    a = p2e(inliers_E[0])
    b = p2e(inliers_E[1])
    plt.plot(a[0], a[1], 'c.')
    plt.plot([a[0], b[0]], [a[1], b[1]], 'c-')
    plt.imshow(img1)
    plt.show()
