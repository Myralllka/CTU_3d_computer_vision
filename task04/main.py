import random

import toolbox
from toolbox import *
import scipy.linalg as lin_alg


def u2H(u1, u2):
    """
    :param u1:  (3×4) the image coordinates of points in the first image (3×4 matrix/np.array)
    :param u2: (3×4) the image coordinates of the corresponding points in the second image.
    :return: H: a 3×3 homography matrix (np.array), or an empty array [] if there is no solution.
    """
    M = list()
    # u1 = u1.T
    # u2 = u2.T

    for i in range(len(u1)):
        m = np.r_[u1[i], [0, 0, 0], -u1[i] * u2[i][0]]
        M.append(m)

    for i in range(len(u1)):
        m = np.r_[[0, 0, 0], u1[i], -u1[i] * u2[i][1]]
        M.append(m)

    H = lin_alg.null_space(M)

    if H.size == 0 or np.linalg.matrix_rank(M) != 8:
        return []
    return (H / H[-1]).reshape(3, 3)


def ransac_H(array_points_1, array_points_2, relation_idxs, iterations=10000):
    indexes = range(len(relation_idxs))
    array_points_1 = e2p(array_points_1.T).T
    array_points_2 = e2p(array_points_2.T).T

    for i in range(iterations):
        current_idxs = relation_idxs[random.sample(indexes, 4)]

        current_pts_1 = array_points_1.T[current_idxs.T[0]]
        current_pts_2 = array_points_2.T[current_idxs.T[1]]

        current_Ha = np.array(u2H(current_pts_1, current_pts_2))
        # print(current_Ha.s)
        if current_Ha.size > 0:
            points_1xHa_2 = current_Ha @ array_points_1
            current_Ha__1 = np.linalg.inv(current_Ha)
            points_2xHa__1_1 = current_Ha__1 @ array_points_2

            for re_idx in relation_idxs:
                print(re_idx)

            # errors_12 = points_1_Ha_2 - array_points_2
            # errors_21 = points_2_Ha__1_1 - array_points_1
            # print(errors_21)


if __name__ == "__main__":
    # Firstly I will start with pair 1 and 2
    fig = plt.figure(1)

    points_book1 = np.loadtxt('task04/data/books_u1.txt').T
    points_book2 = np.loadtxt('task04/data/books_u2.txt').T
    points_1_2_relations = np.loadtxt('task04/data/books_m12.txt', dtype=int)

    book1_img = plt.imread('task04/imgs/book1.png')
    book2_img = plt.imread('task04/imgs/book2.png')
    book1_img = book1_img.copy()
    book2_img = book2_img.copy()

    plt.plot(points_book1[0], points_book1[1], 'r.')
    plt.imshow(book1_img)

    fig = plt.figure(2)
    plt.plot(points_book2[0], points_book2[1], 'r.')

    ransac_H(points_book1, points_book2, points_1_2_relations)
    # plt.imshow(book2_img)
    # plt.show()
