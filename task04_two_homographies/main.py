import matplotlib.pyplot as plt
import numpy as np

import toolbox
from toolbox import *

THETA = 2


def homography_score(H, points_original_plain_euclidean, points_other_plain_euclidean, idxs_relations):
    points_xH = H @ e2p(points_original_plain_euclidean)
    points_xH = p2e(points_xH)
    dist = [np.linalg.norm(points_xH.T[relation[0]] - points_other_plain_euclidean.T[relation[1]])
            for relation in idxs_relations]
    tmp = np.array(dist) < THETA
    return tmp


def ransac_H(points_img_1_euclidean, points_img_2_euclidean, idxs_input_relations, iterations=1000):
    points_img_1_projective = e2p(points_img_1_euclidean)
    points_img_2_projective = e2p(points_img_2_euclidean)

    best_score = 0

    best_inliers_Ha = []
    best_inliers_Hb = []
    best_outliers_Hb = []
    best_a = []

    for i in range(iterations):
        # compute the temporary Ha
        idxs_first_homography = idxs_input_relations[random.sample(range(idxs_input_relations.shape[0]), 4)]

        pts4Ha_img1 = points_img_1_projective.T[idxs_first_homography.T[0]].T
        pts4Ha_img2 = points_img_2_projective.T[idxs_first_homography.T[1]].T

        Ha = u2H(pts4Ha_img1, pts4Ha_img2)
        try:
            Ha_inv = np.linalg.inv(Ha)
        except:
            continue
        # compute Ha inliers

        tmp = homography_score(Ha, points_img_1_euclidean, points_img_2_euclidean, idxs_input_relations)

        idxs_Ha_outliers = idxs_input_relations[~tmp]
        idxs_Ha_inliers = idxs_input_relations[tmp]

        # estimate H
        idxs_second_homography = idxs_Ha_outliers[random.sample(range(idxs_Ha_outliers.shape[0]), 3)]
        pts4H_img1 = points_img_1_projective.T[idxs_second_homography.T[0]].T
        pts4H_img2 = points_img_2_projective.T[idxs_second_homography.T[1]].T

        us = Ha_inv @ pts4H_img2
        us /= us[2]
        us_prime = pts4H_img1

        v = np.cross(np.cross(us[:, 0].T, us_prime[:, 0].T), np.cross(us[:, 1].T, us_prime[:, 1].T))
        # construct matrix A...

        A = np.array([(us_prime[0][0] * v[2] - us_prime[2][0] * v[0]) * us[:, 0].T,
                      (us_prime[0][1] * v[2] - us_prime[2][1] * v[0]) * us[:, 1].T,
                      (us_prime[0][2] * v[2] - us_prime[2][2] * v[0]) * us[:, 2].T])

        ## Now - vector b
        b = np.array([[us[0][0] * us_prime[2][0] - us[2][0] * us_prime[0][0]],
                      [us[0][1] * us_prime[2][1] - us[2][1] * us_prime[0][1]],
                      [us[0][2] * us_prime[2][2] - us[2][2] * us_prime[0][2]]])

        ## a = A.inv @ b
        try:
            a = np.linalg.inv(A) @ b
        except:
            continue
        # Finally  H == I + v @ a.T, Hb = H @ Ha_inv
        H = np.eye(3) + np.outer(v, a)

        Hb = np.linalg.inv(H @ Ha_inv)

        tmp = homography_score(Hb, points_img_1_euclidean, points_img_2_euclidean, idxs_Ha_outliers)

        idxs_Hb_outliers = idxs_Ha_outliers[~tmp]
        idxs_Hb_inliers = idxs_Ha_outliers[tmp]

        score = idxs_Ha_inliers.shape[0] + idxs_Hb_inliers.shape[0]
        if score > best_score:
            best_score = score
            best_a = a
            best_inliers_Ha = idxs_Ha_inliers
            best_inliers_Hb = idxs_Hb_inliers
            best_outliers_Hb = idxs_Hb_outliers

    return best_a, best_inliers_Ha, best_inliers_Hb, best_outliers_Hb


if __name__ == "__main__":
    book1 = 2
    book2 = 3
    points_book1 = np.loadtxt('task04/data/books_u{}.txt'.format(book1)).T
    points_book2 = np.loadtxt('task04/data/books_u{}.txt'.format(book2)).T
    points_1_2_relations = np.loadtxt('task04/data/books_m{}{}.txt'.format(book1, book2), dtype=int)

    img1 = plt.imread('task04/imgs/book{}.png'.format(book1))
    img2 = plt.imread('task04/imgs/book{}.png'.format(book2))
    img1 = img1.copy()
    img2 = img2.copy()

    line, inliers_Ha, inliers_Hb, outliers_Hb = ransac_H(points_book1, points_book2, points_1_2_relations,
                                                         iterations=1000)

    plt.imshow(img1)

    a = points_book1[:, outliers_Hb[:, 0]]
    b = points_book2[:, outliers_Hb[:, 1]]
    plt.plot(a[0], a[1], 'k.', markersize=.8)
    plt.plot([a[0], b[0]], [a[1], b[1]], 'k-', linewidth=.2)

    a = points_book1[:, inliers_Ha[:, 0]]
    b = points_book2[:, inliers_Ha[:, 1]]
    plt.plot(a[0], a[1], 'c.')
    plt.plot([a[0], b[0]], [a[1], b[1]], 'c-')

    a = points_book1[:, inliers_Hb[:, 0]]
    b = points_book2[:, inliers_Hb[:, 1]]
    plt.plot(a[0], a[1], 'r.')
    plt.plot([a[0], b[0]], [a[1], b[1]], 'r-')

    y = np.array([0, 1000])
    x = (-line[2] / line[0]) + (-line[1] / line[0]) * y
    plt.plot(x, y, 'm-', label="ransac", linewidth=2)
    plt.imshow(img1)
    plt.show()
