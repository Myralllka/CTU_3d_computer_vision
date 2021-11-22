import numpy as np  # for matrix computation and linear algebra
import scipy.linalg as lin_alg
import random
import matplotlib.pyplot as plt


def e2p(u_e):
    """
    Transformation of euclidean to projective coordinates
    Synopsis: u_p = e2p( u_e )
    :param u_e: d by n matrix; n euclidean vectors of dimension d
    :return: d+1 by n matrix; n homogeneous vectors of dimension d+1
    """
    return np.vstack((u_e, np.ones(u_e.shape[1])))


def p2e(u_p):
    """
    Transformation of projective to euclidean coordinates
    Synopsis: u_e = p2e( u_p )
    :param u_p: d+1 by n matrix; n homogeneous vectors of dimension d+1
    :return: d by n matrix; n euclidean vectors of dimension d
    """

    u_p /= u_p[-1]
    return u_p[:-1]


def u2H(u1, u2):
    """
    :param u1:  (3×4) the image coordinates of points in the first image (3×4 matrix/np.array)
    :param u2: (3×4) the image coordinates of the corresponding points in the second image.
    :return: H: a 3×3 homography matrix (np.array), or an empty array [] if there is no solution.
    """
    M = list()

    for i in range(u1.shape[1]):
        m = np.r_[u1[:, i], [0, 0, 0], -u1[:, i] * u2[0][i]]
        M.append(m)
        m = np.r_[[0, 0, 0], u1[:, i], -u1[:, i] * u2[1][i]]
        M.append(m)

    M = np.array(M)
    H = lin_alg.null_space(M)

    if H.size == 0 or np.linalg.matrix_rank(M) != 8:
        return []
    return (H / H[-1]).reshape(3, 3)


def sqc(x):
    """
    Skew-symmetric matrix for cross-product
    Synopsis: S = sqc(x)
    :param x: vector 3×1
    :return: skew symmetric matrix (3×3) for cross product with x
    """
    return np.array([[0, x[2], -x[1]], [-x[2], 0, x[0]], [x[1], -x[0], 0]])


def get_intersect_lines(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return float('inf'), float('inf')
    return [x / z, y / z]


def Eu2Rt(E, u1, u2):
    """
    Essential matrix decomposition with cheirality
    Notes: The sessential matrix E is decomposed such that E = R * sqc( b ).

    Synopsis: [R, t] = EutoRt( E, u1, u2 )
    :param E: essential matrix (3×3)
    :param u1, u2: corresponding image points in homogeneous coordinates (3×n), used for cheirality test
    :return: R, t - relative rotation (3×3) or [] if cheirality fails, relative translation, euclidean (3×1), unit length
    """
    # E = R cross C, decomposition:
    Rz = np.array([[0, 1, 0],
                   [-1, 0, 0],
                   [0, 0, 1]])
    U, D, Vt = np.linalg.svd(E)
    R_1 = U @ Rz @ Vt
    R_2 = U @ Rz.T @ Vt
    C_1 = U[:, -1]
    C_2 = -C_1

    P1_c = np.c_[(np.eye(3), np.zeros((3, 1)))]

    result_index_R_C_Ps = []

    for R_loop, C_loop in [[R_1, C_1], [R_1, C_2], [R_2, C_1], [R_2, C_2]]:
        P2_c = np.c_[(R_loop, (R_loop @ C_loop).reshape(3, 1))]
        X = Pu2X(P1_c, P2_c, u1, u2)
        X = p2e(X)
        a = np.logical_and((np.divide((np.sum(np.multiply(X, u1), axis=0)),
                                      (np.multiply((np.linalg.norm(X, axis=0)), (np.linalg.norm(u1, axis=0))))) > 0),
                           (np.divide((np.sum(np.multiply(X, u2), axis=0)),
                                      (np.multiply((np.linalg.norm(X, axis=0)), (np.linalg.norm(u2, axis=0))))) > 0))

        result_index_R_C_Ps.append([np.count_nonzero(a), R_loop, C_loop])

    c, R, C = sorted(result_index_R_C_Ps, key=lambda x: x[0])[-1]
    return [R, C]


def Pu2X(P1, P2, u1, u2):
    """
    Binocular reconstruction by DLT triangulation
    Notes: The essential matrix E is decomposed such that E ~ sqc(t) * R. Note that t = -R*b.

    Synopsis: X = Pu2X( P1, P2, u1, u2 )
    :param P1, P2: projective camera matrices (3×4)
    :param u1, u2: corresponding image points in homogeneous coordinates (3×n)
    :return: X - reconstructed 3D points, homogeneous (4×n)
    """
    res_X = []
    for i in range(len(u1[0])):
        c_u = u1[:, i]
        c_v = u2[:, i]
        D = np.c_[c_u[0] * P1[2] - P1[0],
                  c_u[1] * P1[2] - P2[0],
                  c_v[0] * P2[2] - P1[1],
                  c_v[1] * P2[2] - P2[1]]

        # M = np.vstack([np.c_[c_u, np.zeros((3, 1)), -P1], np.c_[np.zeros((3, 1)), c_v, -P2]])
        # _, _, Vh = np.linalg.svd(M)
        # res_X.append(Vh[-1, 2:] / Vh[-1, -1])
        _, _, Vh = np.linalg.svd(D.T @ D)
        res_X.append(Vh[:, -1] / Vh[-1][-1])
    return np.array(res_X).T


def err_F_sampson(F, u1, u2):
    """
    Sampson error on epipolar geometry

    Synopsis: err = err_F_sampson( F, u1, u2 )
    :param F: fundamental matrix (3×3)
    :param u1, u2: corresponding image points in homogeneous coordinates (3×n)
    :return: e - Squared Sampson error for each correspondence (1×n).
    """
    alg_epipolar_error = err_epipolar(F, u1, u2)
    S = np.array([[1, 0, 0],
                  [0, 1, 0]])
    SF = S @ F
    denom = np.sqrt(np.linalg.norm(SF @ u1, axis=0) ** 2 + np.linalg.norm(SF @ u2, axis=0) ** 2)

    return alg_epipolar_error / denom


def err_epipolar(F, u1, u2):
    """
    compute the epipolar error given fundamental matrix F, u1, u2
    @param F: 3*3 rank2 matrix
    @param u1, u2: 3*n np matrix
    @return: 1*n no matrix
    """
    c_l = F @ u1
    # line normalisation
    # c_l /= np.sqrt(c_l[0] ** 2 + c_l[1] ** 2)
    e = np.abs(np.sum(u2 * c_l, axis=0))
    return e


def err_reprojection(P1, P2, u1, u2, X):
    """
    compute projection error given P1, P2, u1, u2, X

    @param P1, P2: 3*4 np matrix
    @param u1, u2: 3*n np matrix
    @param X: 4*n matrix

    @return: 1*n np matrix
    """
    e1 = P1 @ X
    e1 /= e1[-1]
    e2 = P2 @ X
    e2 /= e2[-1]
    e1 = np.sum(e1 - u1, axis=0)
    e2 = np.sum(e2 - u2, axis=0)
    e1 **= 2
    e2 **= 2
    return e1 + e2


def u_correct_sampson(F, u1, u2):
    """
    Sampson correction of correspondences

    Synopsis: [nu1, nu2] = u_correct_sampson( F, u1, u2 )
    :param F: fundamental matrix (3×3)
    :param u1, u2: corresponding image points in homogeneous coordinates (3×n)
    :return: nu1, nu2 - corrected corresponding points, homog. (3×n).
    """
    alg_epipolar_error = err_epipolar(F, u1, u2)
    S = np.array([[1, 0, 0],
                  [0, 1, 0]])
    SF = S @ F
    denom = np.linalg.norm(SF @ u1, axis=0) ** 2 + np.linalg.norm(SF @ u2, axis=0) ** 2

    fraction = alg_epipolar_error / denom

    # [u1 u2 v1 v2]
    original = np.vstack((p2e(u1), p2e(u2)))
    F1 = F.T[0]
    F2 = F.T[1]
    right = np.vstack((F1 @ u2, F2 @ u2, F1 @ u1, F2 @ u1))
    right2 = fraction * right
    return original - right2


def draw_epipolar_lines(c_u1, c_u2, c_F, img1, img2, header='The epipolar lines using F'):
    colors = ["dimgray", "rosybrown", "maroon", "peru",
              "moccasin", "yellow", "olivedrab", "lightgreen",
              "navy", "royalblue", "indigo", "hotpink"]

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


def ransac_E(c_u1p_K, c_u2p_K, K, theta, optimiser, iterations=1000):
    best_score = 0
    best_R = []
    best_C = []
    best_E = []
    best_idxs = []
    inliers_E, outliers_E = [], []

    K_inv = np.linalg.inv(K)

    c_u1p_K_undone = K_inv @ c_u1p_K
    c_u1p_K_undone /= c_u1p_K_undone[-1]
    c_u2p_K_undone = K_inv @ c_u2p_K
    c_u2p_K_undone /= c_u2p_K_undone[-1]

    for i in range(iterations):
        idxs = random.sample(range(c_u2p_K.shape[1]), 5)
        loop_u1p = c_u1p_K_undone[:, idxs]
        loop_u2p = c_u2p_K_undone[:, idxs]
        Es = optimiser(loop_u1p, loop_u2p)

        for E in Es:
            # e = err_epipolar(K_inv.T @ E @ K_inv, c_u1p_K, c_u2p_K)
            e = err_F_sampson(K_inv.T @ E @ K_inv, c_u1p_K, c_u2p_K)
            u_correct_sampson(K_inv.T @ E @ K_inv, c_u1p_K, c_u2p_K)
            e = e < theta
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
