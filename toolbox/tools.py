import numpy as np
import scipy.linalg as lin_alg
import random
import p5
import scipy.optimize
# implementation of toolbox for tdv course + some other functions
import toolbox.p3p

THETA = 1
THETA2 = 2


class Camera:
    # Camera representation class
    def __init__(self, n: int, img):
        self.n = n
        self.img = img
        self.interest_points_e = None
        self.interest_points_p = None
        self.has_P = False
        self.P = None
        self.K = None
        self.R = None
        self.t = None

    def __hash__(self):
        return hash(self.img)

    def set_P(self, K, R, t):
        assert t is not None, "can not make P, t is None"
        assert K is not None, "can not make P, K is None"
        assert R is not None, "can not make P, R is None"

        self.t = t.reshape(3, )
        self.R = R
        self.K = K
        self.P = self.K @ np.c_[self.R, self.t]
        self.has_P = True


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


def PP2F(P1, P2):
    """

    """
    Q1, q1 = P1[:, :-1], P1[:, -1]
    Q2, q2 = P2[:, :-1], P2[:, -1]
    qq = Q1 @ np.linalg.inv(Q2)
    return qq.T @ sqc(q1 - qq @ q2)


def triangulate_gvg(P1, P2, u1, u2):
    """

    """
    res_X = []
    for i in range(len(u1[0])):
        c_u = u1[:, i]
        c_v = u2[:, i]
        M = np.vstack([np.c_[c_u, np.zeros((3, 1)), -P1], np.c_[np.zeros((3, 1)), c_v, -P2]])
        _, _, Vh = np.linalg.svd(M)
        X = Vh[-1, 2:] / Vh[-1, -1]
        X_trial_1 = P1 @ X
        X_trial_2 = P2 @ X
        if X_trial_1[-1] < 0 or X_trial_2[-1] < 0:
            X = np.array([np.inf, np.inf, np.inf, np.inf])
        res_X.append(X)
    return np.array(res_X).T


def triangulate_3dv(P1, P2, u1, u2):
    """

    """
    res_X = []

    for i in range(u1.shape[1]):
        c_u = u1[:, i]
        c_v = u2[:, i]
        D = np.vstack([c_u[0] * P1[2] - P1[0],
                       c_u[1] * P1[2] - P1[1],
                       c_v[0] * P2[2] - P2[0],
                       c_v[1] * P2[2] - P2[1]])

        _, _, Vt = np.linalg.svd(D)
        X = Vt.T[:, -1]
        res_X.append(X)

    return np.array(res_X).T


def Pu2X_corrected_inliers(P1, P2, u1, u2, corresp):
    """

    """
    F = PP2F(P1, P2)
    theta = THETA2
    u1_basic, u2_basic = u1[:, corresp[0]], u2[:, corresp[1]]
    res_X = triangulate_3dv(P1, P2, u1_basic, u2_basic)
    e = err_reprojection(P1, P2, u1_basic, u2_basic, res_X)
    e = e < theta
    corresp_inliers = corresp[:, e]

    u1_corrected, u2_corrected = u_correct_sampson(F, u1[:, corresp_inliers[0]], u2[:, corresp_inliers[1]])
    res_X = triangulate_3dv(P1, P2, u1_corrected, u2_corrected)
    return res_X, corresp[:, np.nonzero(e)[0]], np.nonzero(e)[0]


def Pu2X_corrected(P1, P2, u1, u2, corresp):
    """
    Pu2X wrapper, that use F for sampson points correction before the reconstruction

    Synopsis: X = Pu2X( P1, P2, u1, u2, F, corresp)
    @param P1, P2: projective camera matrices (3×4)
    @param u1, u2: corresponding image points in homogeneous coordinates (3×n)
    @param corresp: image correspondences for u1 u2
    @return: X - reconstructed 3D points, homogeneous (4×n)
    """
    F = PP2F(P1, P2)

    u1_corrected, u2_corrected = u_correct_sampson(F, u1[:, corresp[0]], u2[:, corresp[1]])

    # return triangulate_3dv(P1, P2, u1_corrected, u2_corrected)
    return triangulate_3dv(P1, P2, u1_corrected, u2_corrected)


def err_F_sampson(F, u1, u2):
    """
    Sampson error on epipolar geometry

    Synopsis: err = err_F_sampson( F, u1, u2 )
    :param F: fundamental matrix (3×3)
    :param u1, u2: corresponding image points in homogeneous coordinates (3×n)
    :return: e - Squared Sampson error for each correspondence (1×n).
    """
    # alg_epipolar_error = err_epipolar(F, u1, u2)

    S = np.array([[1, 0, 0],
                  [0, 1, 0]])
    Fu1 = F @ u1
    denom = np.linalg.norm(S @ Fu1, axis=0) ** 2 + np.linalg.norm(S @ F.T @ u2, axis=0) ** 2
    return (np.sum(Fu1 * u2, axis=0) ** 2) / denom


def err_epipolar(F, u1, u2):
    """
    compute the epipolar error given fundamental matrix F, u1, u2
    @param F: 3*3 rank2 matrix
    @param u1, u2: 3*n np matrix
    @return: 1*n no matrix
    """
    return np.sum((F @ u1) * u2, axis=0)


def err_reprojection(P1, P2, u1, u2, X):
    """
    compute projection error given P1, P2, u1, u2, X

    @param P1, P2: 3*4 np matrix
    @param u1, u2: 3*n np matrix
    @param X: 4*n matrix

    @return: 1*n np matrix
    """
    e1 = err_reprojection_half(P1, X, u1)
    e2 = err_reprojection_half(P2, X, u2)

    return e1 + e2


def err_reprojection_half(P, X, u):
    """
    compute projection error given P, X, u

    :param P: 3*4 np matrix
    :param X: 4*n matrix
    :param u: 3*n np matrix

    :return: 1*n np matrix
    """
    e = P @ X
    e /= e[-1]
    e = (e - u) ** 2
    e = np.sum(e, axis=0)
    return np.sqrt(e)


def u_correct_sampson(F, u1, u2):
    """
    Sampson correction of correspondences

    Synopsis: [nu1, nu2] = u_correct_sampson( F, u1, u2 )
    :param F: fundamental matrix (3×3)
    :param u1, u2: corresponding image points in homogeneous coordinates (3×n)
    :return: nu1, nu2 - corrected corresponding points, homog. (3×n).
    """
    # return u1, u2
    # alg_epipolar_error = err_epipolar(F, u1, u2)

    Fx = F @ u1
    alg_epipolar_error = np.sum(Fx * u2, axis=0)

    S = np.array([[1, 0, 0],
                  [0, 1, 0]])
    denom = np.linalg.norm(S @ Fx, axis=0) ** 2 + np.linalg.norm(S @ F.T @ u2, axis=0) ** 2
    fraction = alg_epipolar_error / denom

    # [u1 u2 v1 v2]
    original = np.vstack((p2e(u1), p2e(u2)))
    F1 = F[0]
    F2 = F[1]
    right = np.vstack((F1 @ u2, F2 @ u2, F1 @ u1, F2 @ u1))
    right2 = fraction * right
    res = original - right2
    return e2p(res[:2]), e2p(res[2:])


def ransac_Rt_p3p(c_X, c_up_K, corresp_Xu, c_K):
    s = 3
    P = 0.9999
    eps = 0.1
    theta = THETA2
    ###

    best_score = 0
    best_R, best_t = [], []
    inliers_corresp_idxs = []
    counter = 0
    N = np.inf
    for i in range(400):
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
                inliers_corresp_idxs = np.nonzero(e)
                eps += np.count_nonzero(e) / corresp_Xu.shape[1]
                N = np.log(1 - P) / np.log(1 - eps ** s)
                # print(best_score)
        counter += 1
    inliers_idxs = corresp_Xu[:, inliers_corresp_idxs[0]]
    # return best_R, best_t, inliers_idxs, inliers_corresp_idxs[0]

    input_rotation_t = np.concatenate((R2mrp(best_R), best_t.reshape(3, )))

    res = scipy.optimize.fmin(Rt_minimisation_function_X2im,
                              input_rotation_t,
                              (c_X, c_up_K, inliers_idxs, c_K),
                              xtol=10e-100)
    n_R = mrp2R(res[0:3])
    n_t = res[3:]
    # make scale eq to 1
    # n_t /= np.sqrt(n_t[0] ** 2 + n_t[1] ** 2 + n_t[2] ** 2)

    return n_R, n_t, inliers_idxs, inliers_corresp_idxs[0]


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
    P = 0.9999
    eps = 0.5

    ###

    best_score = 0
    best_R, best_t, best_E = [], [], []
    inliers_E_idxs = []

    K_inv = np.linalg.inv(K)

    c_u1p_K_undone = K_inv @ c_u1p_K
    c_u1p_K_undone /= c_u1p_K_undone[-1]
    c_u2p_K_undone = K_inv @ c_u2p_K
    c_u2p_K_undone /= c_u2p_K_undone[-1]

    counter = 1
    N = np.log(1 - P) / np.log(1 - eps ** s)
    for i in range(iterations):
        # while counter <= N:
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
                eps = np.count_nonzero(e) / correspondences.shape[1]
                N = np.log(1 - P) / np.log(1 - eps ** s)
                # print(best_score)
                # print(counter, N)
            counter += 1

    return best_E, best_R, best_t, inliers_E_idxs[0]


def Rt_minimisation_function_im2im(vector, m_u1p, m_u2p, corres, m_K):
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


def Rt_minimisation_function_X2im(vector, m_Xp, m_u2p, corres, m_K):
    """
    function to minimise R and t
    from correspondent points and calibration matrix
    @param vector: [0:3] rotation
    @param vector: [3:] translation
    """
    l_Xp = m_Xp[:, corres[0]]
    l_u2p = m_u2p[:, corres[1]]

    R = mrp2R(vector[0:3])
    P = m_K @ np.c_[R, vector[3:].reshape(3, )]

    e = err_reprojection_half(P, l_Xp, l_u2p)

    return np.sum(e)


def u2ERt_optimal(u1p_K, u2p_K, corresp, K, THETA=1, solver=p5.p5gb, iterations=1000):
    """
    ransac_ERt_inliers wrapper, that also make an optimisation of R and t
    """
    E, R, t, inliers_corresp_idxs = ransac_ERt_inliers(u1p_K, u2p_K, corresp, K, THETA, solver, iterations)
    inliers_idxs = corresp[:, inliers_corresp_idxs]

    input_rotation_t = np.concatenate((R2mrp(R), t))

    res = scipy.optimize.fmin(Rt_minimisation_function_im2im,
                              input_rotation_t,
                              (u1p_K, u2p_K, inliers_idxs, K),
                              xtol=10e-10)
    n_R = mrp2R(res[0:3])
    n_t = res[3:]
    # make scale eq to 1
    n_t /= np.sqrt(n_t[0] ** 2 + n_t[1] ** 2 + n_t[2] ** 2)

    n_E = sqc(-n_t) @ n_R
    return n_E, n_R, n_t, inliers_idxs, inliers_corresp_idxs
