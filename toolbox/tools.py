import numpy as np
import scipy.linalg as lin_alg
import random
import matplotlib.pyplot as plt
import p5
import scipy.optimize


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
    return np.array([[0, x[2], -x[1]],
                     [-x[2], 0, x[0]],
                     [x[1], -x[0], 0]])


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


def R2mrp(R):
    """
    Rotation matrix (9 parameters) to rotation vector (3 parameters) using rodrigues formula
    source: https://courses.cs.duke.edu/fall13/compsci527/notes/rodrigues.pdf
    @param R: rotation matrix
    @return: rotation vector
    """
    assert np.isclose(1, np.linalg.det(R), atol=1e-05, equal_nan=False), "det(R) should be 1"

    A = (R - R.T) / 2
    ro = [A[2, 1], A[0, 2], A[1, 0]]
    s = np.linalg.norm(ro)
    c = (np.sum(np.diag(R)) - 1) / 2
    if np.isclose(s, 0, atol=1e-04) and np.isclose(c, 1, atol=1e-5):
        return [0, 0, 0]
    if np.isclose(s, 0, atol=1e-04) and np.isclose(c, -1, atol=1e-5):
        # TODO
        assert "todo"
        RI = R + np.eye(3)
        # any nonzero column
        u = RI[:, 0] / np.linalg.norm(RI[:, 0])
        return np.pi * u
    u = ro / s
    theta = np.arctan2(s, c)
    return u * theta


def mrp2R(r):
    """
    modified rodrigues parameters to matrix
     source: https://courses.cs.duke.edu/fall13/compsci527/notes/rodrigues.pdf
    @param r: rotation vector
    @return: rotation matrix
    """
    # assert np.linalg.norm(r) < np.pi, "norm od r should be less than pi"
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    u = r / theta
    return np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * np.outer(u, u) + sqc(u) * np.sin(theta)


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
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    R_1 = U @ Rz @ Vt
    R_2 = U @ Rz.T @ Vt
    t_1 = U[:, -1]
    t_2 = -t_1

    P1_c = np.c_[(np.eye(3), np.zeros((3, 1)))]

    result_index_R_C_Ps = []

    for R_loop, t_loop in [[R_1, t_1], [R_1, t_2], [R_2, t_1], [R_2, t_2]]:
        P2_c = np.c_[(R_loop, t_loop.reshape(3, 1))]
        Xs = Pu2X(P1_c, P2_c, u1, u2)
        Xs = p2e(Xs)
        if np.any(Xs[-1] < 0):
            result_index_R_C_Ps.append([-100, R_loop, t_loop])
            continue
        tmp = P2_c @ e2p(Xs)
        if np.any(tmp[:, 2] < 0):
            result_index_R_C_Ps.append([-100, R_loop, t_loop])
            continue
        # a = np.logical_and((np.divide((np.sum(np.multiply(Xs, u1), axis=0)),
        #                               (np.multiply((np.linalg.norm(Xs, axis=0)), (np.linalg.norm(u1, axis=0))))) > 0),
        #                    (np.divide((np.sum(np.multiply(Xs, u2), axis=0)),
        #                               (np.multiply((np.linalg.norm(Xs, axis=0)), (np.linalg.norm(u2, axis=0))))) > 0))

        # result_index_R_C_Ps.append([np.count_nonzero(a), R_loop, t_loop, Xs])
        result_index_R_C_Ps.append([5, R_loop, t_loop])
    c, R, t = sorted(result_index_R_C_Ps, key=lambda x: x[0])[-1]
    if c != u1.shape[1]:
        return None, None
    return [R, t]


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
        # D = np.c_[c_u[0] * P1[2] - P1[0],
        #           c_u[1] * P1[2] - P1[1],
        #           c_v[0] * P2[2] - P2[0],
        #           c_v[1] * P2[2] - P2[1]]

        M = np.vstack([np.c_[c_u, np.zeros((3, 1)), -P1], np.c_[np.zeros((3, 1)), c_v, -P2]])
        _, _, Vh = np.linalg.svd(M)
        res_X.append(Vh[-1, 2:] / Vh[-1, -1])

        # _, _, Vh = np.linalg.svd(D)
        # x = Vh.T[:, -1]
        # x /= x[-1]
        # res_X.append(x)
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
    t = np.einsum("ij,ij->i", u2.T, (F @ u1).T).T
    return np.abs(t)


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
    res = original - right2
    return e2p(res[:2]), e2p(res[2:])


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


def ransac_ERt_inliers(c_u1p_K, c_u2p_K, correspondences, K, theta, optimiser, iterations=1000):
    best_score = 0
    best_R, best_t, best_E = [], [], []
    inliers_E_idxs = []

    K_inv = np.linalg.inv(K)

    c_u1p_K_undone = K_inv @ c_u1p_K
    c_u1p_K_undone /= c_u1p_K_undone[-1]
    c_u2p_K_undone = K_inv @ c_u2p_K
    c_u2p_K_undone /= c_u2p_K_undone[-1]

    for i in range(iterations):
        corresp_idxs = random.sample(range(correspondences.shape[1]), 5)
        corresp_idxs = correspondences[:, corresp_idxs]
        loop_u1p = c_u1p_K_undone[:, corresp_idxs[0]]
        loop_u2p = c_u2p_K_undone[:, corresp_idxs[1]]
        Es = optimiser(loop_u1p, loop_u2p)
        for E in Es:
            F = K_inv.T @ E @ K_inv
            e = err_F_sampson(F, c_u1p_K[:, correspondences[0]], c_u2p_K[:, correspondences[1]])
            e = [(1 - (el ** 2) / (theta ** 2)) if el < theta else 0 for el in e]
            if np.sum(e) > best_score:
                R_c, t_c = Eu2Rt(E, loop_u1p, loop_u1p)
                if R_c is None:
                    continue
                best_score = np.sum(e)
                best_R = R_c
                best_t = t_c
                best_E = E
                inliers_E_idxs = np.nonzero(e)
                print(best_score)
    return best_E, best_R, best_t, inliers_E_idxs[0]


def minimisation_function(vector, m_u1p, m_u2p, corres, m_K):
    """
    function to minimise

    @param vector: [0:3] rotation
    @param vector: [3:] translation
    """
    l_u1p = m_u1p[:, corres[0]]
    l_u2p = m_u2p[:, corres[1]]

    R = mrp2R(vector[0:3])
    E = sqc(- vector[3:]) @ R

    kinv = np.linalg.inv(m_K)
    e = err_F_sampson(kinv.T @ E @ kinv, l_u1p, l_u2p)
    return np.sum(e)


def u2ERt_optimal(u1p_K, u2p_K, corresp, K, THETA=1, solver=p5.p5gb, iterations=1000):
    E, R, t, inliers_corresp_idxs = ransac_ERt_inliers(u1p_K, u2p_K, corresp, K, THETA, solver, iterations)
    inliers_idxs = corresp[:, inliers_corresp_idxs]

    return E, R, t, inliers_idxs, inliers_corresp_idxs
    #
    # input_rotation_t = np.concatenate((R2mrp(R), t))
    # res = scipy.optimize.fmin(minimisation_function,
    #                           input_rotation_t,
    #                           (u1p_K, u2p_K, inliers_idxs, K),
    #                           xtol=10e-10)
    # n_R = mrp2R(res[0:3])
    # n_t = res[3:]
    # new_E = sqc(-n_t) @ n_R
    # return new_E, n_R, n_t, inliers_idxs, inliers_corresp_idxs


def plot_csystem(ax, base, origin, name, color=None):
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

    if color is None:
        colors = ["red", "green", "blue"]
    else:
        colors = [color] * 3
    ax.quiver3D(origin[0], origin[1], origin[2],
                base[0, 0], base[1, 0], base[2, 0],
                length=1,
                arrow_length_ratio=0.1,
                color=colors[0])
    ax.quiver3D(origin[0], origin[1], origin[2],
                base[0, 1], base[1, 1], base[2, 1],
                length=1,
                arrow_length_ratio=0.1,
                color=colors[1])
    ax.quiver3D(origin[0], origin[1], origin[2],
                base[0, 2], base[1, 2], base[2, 2],
                length=1,
                arrow_length_ratio=0.1,
                color=colors[2])
    ax.text(base[0, 0], base[1, 0], base[2, 0], name + "_x")
    ax.text(base[0, 1], base[1, 1], base[2, 1], name + "_y")
    ax.text(base[0, 2], base[1, 2], base[2, 2], name + "_z")
