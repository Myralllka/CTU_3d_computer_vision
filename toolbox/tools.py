import numpy as np
import scipy.linalg as lin_alg
import random
import p5
import scipy.optimize

# implementation of toolbox for tdv course + some other functions
import toolbox.p3p


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
    return np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * np.outer(u, u) - sqc(u) * np.sin(theta)


def Eu2Rt(E, u1, u2):
    """
    Essential matrix decomposition with cheirality
    Notes: The sessential matrix E is decomposed such that E = R * sqc( b ).

    Synopsis: [R, t] = EutoRt( E, u1, u2 )
    @param E: essential matrix (3×3)
    @param u1, u2: corresponding image points in homogeneous coordinates (3×n), used for cheirality test
    @return: R, t - relative rotation (3×3) or [None, None] if cheirality fails, relative translation, euclidean (3×1), unit length
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
        M = np.vstack([np.c_[c_u, np.zeros((3, 1)), -P1], np.c_[np.zeros((3, 1)), c_v, -P2]])
        _, _, Vh = np.linalg.svd(M)
        res_X.append(Vh[-1, 2:] / Vh[-1, -1])
    return np.array(res_X).T


def Pu2X_optimised(P1, P2, u1, u2, F, corresp):
    """
    Pu2X wrapper, that use F for sampson points correction before the reconstruction

    Synopsis: X = Pu2X( P1, P2, u1, u2, F, corresp)
    @param P1, P2: projective camera matrices (3×4)
    @param u1, u2: corresponding image points in homogeneous coordinates (3×n)
    @param F: fundamental matrix
    @param corresp: image correspondences for u1 u2
    @return: X - reconstructed 3D points, homogeneous (4×n)
    """
    res_X = []
    u1_corrected, u2_corrected = u_correct_sampson(F, u1[:, corresp[0]], u2[:, corresp[1]])

    for i in range(len(u1_corrected[0])):
        c_u = u1_corrected[:, i]
        c_v = u2_corrected[:, i]
        M = np.vstack([np.c_[c_u, np.zeros((3, 1)), -P1], np.c_[np.zeros((3, 1)), c_v, -P2]])
        _, _, Vh = np.linalg.svd(M)
        res_X.append(Vh[-1, 2:] / Vh[-1, -1])
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

    denom = np.sqrt(np.linalg.norm(S @ F @ u1, axis=0) ** 2 +
                    np.linalg.norm(S @ F.T @ u2, axis=0) ** 2)

    return alg_epipolar_error / denom


def err_epipolar(F, u1, u2):
    """
    compute the epipolar error given fundamental matrix F, u1, u2
    @param F: 3*3 rank2 matrix
    @param u1, u2: 3*n np matrix
    @return: 1*n no matrix
    """
    return np.abs(np.sum((F @ u1) * u2, axis=0))


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


def err_reprojection_half(P, X, u):
    """
    compute projection error given P, u, X

    @param P: 3*4 np matrix
    @param u: 3*n np matrix
    @param X: 4*n matrix

    @return: 1*n np matrix
    """
    e = P @ X
    e /= e[-1]
    return np.abs(np.sum(e - u, axis=0))


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


def ransac_Rt_p3p(c_X, c_up_K, corresp_Xu, c_K):
    s = 3
    P = 0.999
    eps = 0.1
    theta = 100
    ###

    best_score = 0
    best_R, best_t = [], []
    inliers_idxs = []
    counter = 0
    c_X = e2p(c_X)
    N = np.log(1 - P) / np.log(1 - eps ** s)
    for i in range(300):
        # while counter <= N:
        corresp_idxs = random.sample(range(corresp_Xu.shape[1]), s)
        corresp_idxs = corresp_Xu[:, corresp_idxs]
        loop_X = c_X[:, corresp_idxs[0]]
        loop_up = c_up_K[:, corresp_idxs[1]]
        loop_up_K_undone = np.linalg.inv(c_K) @ loop_up
        loop_up_K_undone /= loop_up_K_undone[-1]
        l_X_new_arr = toolbox.p3p.p3p_grunert(loop_X, np.linalg.inv(c_K) @ loop_up)
        for l_X_new in l_X_new_arr:
            n_R, n_t = toolbox.p3p.XX2Rt_simple(loop_X, l_X_new)
            l_P = c_K @ np.c_[n_R, n_t]

            e = err_reprojection_half(l_P, c_X[:, corresp_Xu[0]], c_up_K[:, corresp_Xu[1]])
            e = [(1 - (el ** 2) / (theta ** 2)) if el < theta else 0 for el in e]
            if np.sum(e) > best_score:
                best_score = np.sum(e)
                best_R = n_R
                best_t = n_t
                inliers_idxs = np.nonzero(e)
                eps += np.count_nonzero(e) / corresp_Xu.shape[1]
                N = np.log(1 - P) / np.log(1 - eps ** s)
                print(best_score)
        counter += 1

    return best_R, best_t, inliers_idxs[0]


def ransac_ERt_inliers(c_u1p_K, c_u2p_K, correspondences, K, theta, essential_matrix_estimator, iterations=1000):
    """
    Find E, R, t and respective inliers given points correspondences with RANSAC method.
    @param c_u1p_K, c_u2p_K: 3xn matrices of 2d points from images in homogenous coordinate system, with K applied
    @param correspondences: 2xn list of correspondent indexes for c_u1p_K, c_u2p_K
    @param K: Camera calibration matrix
    @param theta: threshold for ransac to accept inliers, in pxs
    @param essential_matrix_estimator: function that estimate set of essential matrices given 5 correspondences
    @param iterations: number of iterations for ransac
    @return: E, R, t, inliers
    """
    # P     -- probability that the last proposal is all-inlier
    # eps   -- the fraction of inliers among primitives
    # s     -- minimal configuration size (2 for line, 5 for E estimation)

    s = 5
    P = 0.99
    eps = 0.1

    ###

    best_score = 0
    best_R, best_t, best_E = [], [], []
    inliers_E_idxs = []

    K_inv = np.linalg.inv(K)

    c_u1p_K_undone = K_inv @ c_u1p_K
    c_u1p_K_undone /= c_u1p_K_undone[-1]
    c_u2p_K_undone = K_inv @ c_u2p_K
    c_u2p_K_undone /= c_u2p_K_undone[-1]

    counter = 0
    N = np.log(1 - P) / np.log(1 - eps ** s)
    while counter <= N:
        # for i in range(iterations):
        corresp_idxs = random.sample(range(correspondences.shape[1]), 5)
        corresp_idxs = correspondences[:, corresp_idxs]
        loop_u1p = c_u1p_K_undone[:, corresp_idxs[0]]
        loop_u2p = c_u2p_K_undone[:, corresp_idxs[1]]
        Es = essential_matrix_estimator(loop_u1p, loop_u2p)
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
                eps += np.count_nonzero(e) / correspondences.shape[1]
                N = np.log(1 - P) / np.log(1 - eps ** s)
                print(best_score)
                print(counter, N)
            counter += 1

    return best_E, best_R, best_t, inliers_E_idxs[0]


def Rt_minimisation_function(vector, m_u1p, m_u2p, corres, m_K):
    """
    function to minimise R and t
    from correspondent points and calibration matrix
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
    """
    ransac_ERt_inliers wrapper, that also make an optimisation of R and t
    """
    E, R, t, inliers_corresp_idxs = ransac_ERt_inliers(u1p_K, u2p_K, corresp, K, THETA, solver, iterations)
    inliers_idxs = corresp[:, inliers_corresp_idxs]

    input_rotation_t = np.concatenate((R2mrp(R), t))

    res = scipy.optimize.fmin(Rt_minimisation_function,
                              input_rotation_t,
                              (u1p_K, u2p_K, inliers_idxs, K),
                              xtol=10e-10)
    n_R = mrp2R(res[0:3])
    n_t = res[3:]
    # make scale eq to 1
    n_t /= np.sqrt(n_t[0] ** 2 + n_t[1] ** 2 + n_t[2] ** 2)

    n_E = sqc(-n_t) @ n_R
    return n_E, n_R, n_t, inliers_idxs, inliers_corresp_idxs
