import toolbox
from toolbox import *

if __name__ == "__main__":
    ### Preparing, loading the data
    view_1 = 1
    view_2 = 2

    points_view_1 = np.loadtxt('task_general/data/u_{:02}.txt'.format(view_1)).T
    points_view_2 = np.loadtxt('task_general/data/u_{:02}.txt'.format(view_2)).T
    points_1_2_relations = np.loadtxt('task_general/data/m_{:02}_{:02}.txt'.format(view_1, view_2), dtype=int).T

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
    u1p_K_undone = K_inv @ u1p_K
    u1p_K_undone /= u1p_K_undone[-1]
    u2p_K_undone = K_inv @ u2p_K
    u2p_K_undone /= u2p_K_undone[-1]

    ### TODO: HERE ransac start

    Es = p5.p5gb(u1p_K, u2p_K)

    for E in Es:
        R_c, t_c = Eu2Rt(E, u1p_K, u2p_K)
        F = K_inv.T @ E @ K_inv
        e = err_F_sampson(F, u1p_K, u2p_K)
