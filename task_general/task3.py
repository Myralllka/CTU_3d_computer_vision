import toolbox
from toolbox import *

THETA = 0.5

if __name__ == "__main__":
    ### Preparing, loading the data
    view_1 = 1
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

    # compute sampson error
    # optimize

    # F = K_inv.T @ E @ K_inv
    #
    # draw_epipolar_lines(inliers_E[0], inliers_E[1], F, img1, img2)

    # TODO: use scipy.optimize.fmin
    # TODO: use rodrigues rotation formula
    # TODO: rewrite ransac to return inliers indexes

    # Use reprojection errors
