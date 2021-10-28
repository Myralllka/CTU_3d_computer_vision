import numpy as np  # for matrix computation and linear algebra
import scipy.linalg as lin_alg


def e2p(u_e):
    """
    Transformation of euclidean to projective coordinates
    Synopsis: u_p = e2p( u_e )
    :param u_e: d by n matrix; n euclidean vectors of dimension d
    :return: d+1 by n matrix; n homogeneous vectors of dimension d+1
    """
    return np.vstack((u_e, np.ones(u_e.shape[1])))
    # For tasks before 0.4
    # return np.c_[u_e, np.ones(len(u_e))]


def p2e(u_p):
    """
    Transformation of projective to euclidean coordinates
    Synopsis: u_e = p2e( u_p )
    :param u_p: d+1 by n matrix; n homogeneous vectors of dimension d+1
    :return: d by n matrix; n euclidean vectors of dimension d
    """

    u_p /= u_p[2]
    return u_p[:-1]

    # For tasks before 0.4
    # u_p = np.array(list(map(lambda x: np.array([x[0] / x[-1], x[1] / x[-1], 1]), u_p)))
    # return np.delete(u_p, -1, axis=1)


def u2H(u1, u2):
    """
    :param u1:  (3×4) the image coordinates of points in the first image (3×4 matrix/np.array)
    :param u2: (3×4) the image coordinates of the corresponding points in the second image.
    :return: H: a 3×3 homography matrix (np.array), or an empty array [] if there is no solution.
    """
    M = list()
    # u1 = u1.T
    # u2 = u2.T

    for i in range(u1.shape[1]):
        m = np.r_[u1[:, i], [0, 0, 0], -u1[:, i] * u2[0][i]]
        M.append(m)
        m = np.r_[[0, 0, 0], u1[:, i], -u1[:, i] * u2[1][i]]
        M.append(m)

    # for i in range(len(u1)):
    #     m = np.r_[[0, 0, 0], u1[:, i], -u1[:, i] * u2[1][i]]
    #     M.append(m)
    M = np.array(M)
    H = lin_alg.null_space(M)

    if H.size == 0 or np.linalg.matrix_rank(M) != 8:
        return []
    return (H / H[-1]).reshape(3, 3)


def vlen(x):
    """
    Column vectors length
    Synopsis: l = vlen( x )
    :param x: d by n matrix; n vectors of dimension d
    :return: 1 by n row vector; euclidean lengths of the vectors
    """
    # def sqr_of_vec(vec):
    #     a = lambda x: x**2
    return np.array(list(map(lambda z: np.sqrt(sum(list(map(lambda y: y ** 2, z)))), x)))


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


def EutoRt(E, u1, u2):
    """
    Essential matrix decomposition with cheirality
    Notes: The sessential matrix E is decomposed such that E = R * sqc( b ).

    Synopsis: [R, t] = EutoRt( E, u1, u2 )
    :param E: essential matrix (3×3)
    :param u1, u2: corresponding image points in homogeneous coordinates (3×n), used for cheirality test
    :return: R, t - relative rotation (3×3) or [] if cheirality fails, relative translation, euclidean (3×1), unit length
    """
    pass


def Pu2X(P1, P2, u1, u2):
    """
    Binocular reconstruction by DLT triangulation
    Notes: The sessential matrix E is decomposed such that E ~ sqc(t) * R. Note that t = -R*b.

    Synopsis: X = Pu2X( P1, P2, u1, u2 )
    :param P1, P2: projective camera matrices (3×4)
    :param u1, u2: corresponding image points in homogeneous coordinates (3×n)
    :return: X - reconstructed 3D points, homogeneous (4×n)
    """
    pass


def err_F_sampson(F, u1, u2):
    """
    Sampson error on epipolar geometry

    Synopsis: err = err_F_sampson( F, u1, u2 )
    :param F: fundamental matrix (3×3)
    :param u1, u2: corresponding image points in homogeneous coordinates (3×n)
    :return: e - Squared Sampson error for each correspondence (1×n).
    """
    pass


def u_correct_sampson(F, u1, u2):
    """
    Sampson correction of correspondences

    Synopsis: [nu1, nu2] = u_correct_sampson( F, u1, u2 )
    :param F: fundamental matrix (3×3)
    :param u1, u2: corresponding image points in homogeneous coordinates (3×n)
    :return: nu1, nu2 - corrected corresponding points, homog. (3×n).
    """
    pass
