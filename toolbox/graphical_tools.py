import matplotlib.pyplot as plt
import random
import numpy as np


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


def draw_epipolar_lines(c_u1, c_u2, c_F, img1, img2, header='The epipolar lines using F'):
    """
    Draw epipolar lines for pair of images. The number of lines = size of 'colors' array of this function
    @param c_u1, c_u2: 2d points in homogenous coordinate system, 3xn matrices
    @param c_F: Fundamental matrix
    @param img1, img2: images for c_u1, c_u2 respectively
    @param header: the header for the plot
    """
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
