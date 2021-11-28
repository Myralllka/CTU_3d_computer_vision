import toolbox
from toolbox import *

THETA = 1

if __name__ == "__main__":
    ### Preparing, loading the data
    view_1 = 1
    view_2 = 2

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

    E, R, C, inliers_idxs, inliers_corresp_idxs = u2ERC_optimal(u1p_K, u2p_K, points_1_2_relations, K)

    draw_epipolar_lines(u1p_K[:, inliers_idxs[0]], u2p_K[:, inliers_idxs[1]], K_inv.T @ E @ K_inv, img1, img2)

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
