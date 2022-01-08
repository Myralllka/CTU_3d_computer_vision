import sys

import toolbox
from toolbox import *
import scipy.io
import matlab.engine

flab = matlab.engine.start_matlab()

NUM_OF_IMGS = 12

random.seed(1)

if __name__ == "__main__":
    ### Preparing, loading the data
    ### cameras located like this:
    ### 1--2--3--4
    ### |  |  |  |
    ### 5--6--7--8
    ### |  |  |  |
    ### 9-10-11-12
    ### numeration from 0, so in the code they will be:
    ### 0--1--2--3
    ### |  |  |  |
    ### 4--5--6--7
    ### |  |  |  |
    ### 8--9-10-11
    ### So the best choice for the very beginning is the pair 6--7
    ### Data setup

    # imgs_order = [7, 11, 3, 6, 10, 2, 5, 9, 1, 4, 8, 0]
    # imgs_order = [5, 6, 0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
    pairs = ((7, 11),
             (0, 1), (0, 4), (4, 5), (4, 8), (8, 9),
             (1, 2), (1, 5), (5, 6), (5, 9), (9, 10),
             (2, 3), (2, 6), (6, 7), (6, 10), (10, 11),
             (3, 7))

    imgs_order = [7, 11]
    cameras = dict()
    Es = []

    # init K
    K = np.loadtxt('task_general/K.txt')
    K_inv = np.linalg.inv(K)

    # init images and cameras
    for i in range(NUM_OF_IMGS):
        img = plt.imread('task_general/imgs/{:02}.jpg'.format(i + 1))
        cameras[i] = Camera(i, deepcopy(img))

    c = Corresp(NUM_OF_IMGS)

    # for debug info
    c.verbose = 0

    # construct images correspondences
    for view_1 in range(NUM_OF_IMGS):
        cameras[view_1].interest_points_e = np.loadtxt(
                'task_general/data/u_{:02}.txt'.format(view_1 + 1)).T
        cameras[view_1].interest_points_p = e2p(
                cameras[view_1].interest_points_e)
        for view_2 in range(view_1 + 1, NUM_OF_IMGS):
            points_relations = np.loadtxt(
                    'task_general/data/m_{:02}_{:02}.txt'.format(view_1 + 1,
                                                                 view_2 + 1),
                    dtype=int)
            c.add_pair(view_1, view_2, points_relations)

    idx_cam1 = imgs_order[0]
    idx_cam2 = imgs_order[1]
    print("[task4] adding camera {}".format(idx_cam1))
    print("[task4] adding camera {}".format(idx_cam2))
    u1p_K = cameras[idx_cam1].interest_points_p
    u2p_K = cameras[idx_cam2].interest_points_p

    corresp_u2u = np.array(c.get_m(idx_cam1, idx_cam2))

    #  initial pair reconstruction
    E, R, t, corresp_u2u_inliers, corresp_u2u_inliers_idxs = u2ERt_optimal(
            u1p_K, u2p_K, corresp_u2u, K)
    Es.append(E)
    F = K_inv.T @ (sqc(-t) @ R) @ K_inv

    cameras[idx_cam1].set_P(K, np.eye(3), np.zeros((3, 1)))
    cameras[idx_cam2].set_P(K, R, t)

    X = p2e(Pu2X_corrected(cameras[idx_cam1].P,
                           cameras[idx_cam2].P,
                           u1p_K,
                           u2p_K,
                           corresp_u2u_inliers))
    # init 3d points
    # ax.plot3D(X[0], X[1], X[2], 'b,')
    c.start(imgs_order[0], imgs_order[1], corresp_u2u_inliers_idxs)
    #  add one more camera
    new_Xs = []
    # add all other cameras
    # Compute Rs and ts!
    for k in range(NUM_OF_IMGS - 2):
        ig = np.array(
                sorted(list(np.array([i for i in c.get_green_cameras()]).T),
                       key=lambda x: x[1], reverse=True))
        # Xucount = c.get_Xucount(3)
        i = ig[0][0]
        imgs_order.append(i)
        print("[task4] adding camera {}".format(i))
        corresp_X2u = np.array(c.get_Xu(i))[:-1]

        R2, t2, corresp_X2u_inliers, corresp_X2u_inliers_idxs = ransac_Rt_p3p(
                e2p(X),
                e2p(cameras[i].interest_points_e),
                corresp_X2u,
                K)
        cameras[i].set_P(K, R2, t2)

        c.join_camera(i, corresp_X2u_inliers_idxs)

        ilist = c.get_cneighbours(i)
        P1 = cameras[i].P
        for ic in ilist:
            corresp_u2u_idxs = np.array(c.get_m(i, ic))
            P2 = cameras[ic].P

            R21 = cameras[ic].R @ cameras[i].R.T
            t21 = cameras[ic].t - (R21 @ cameras[i].t)
            F = K_inv.T @ sqc(-t21) @ R21 @ K_inv

            new_Xs, corresp_Xs_inliers, corresp_Xs_inliers_idxs = Pu2X_corrected_inliers(
                    P1,
                    P2,
                    cameras[i].interest_points_p,
                    cameras[ic].interest_points_p,
                    corresp_u2u_idxs)
            if not new_Xs.size == 0:
                c.new_x(i, ic, corresp_Xs_inliers_idxs)
                X = np.append(X, p2e(new_Xs), axis=1)
        ilist = c.get_selected_cameras()
        for ic in ilist:
            corr_ok = []
            [X_i, u_i, Xu_verified] = c.get_Xu(ic)
            l_corresp_X2u = np.array([X_i, u_i])
            for loop_index in range(l_corresp_X2u.shape[1]):
                if not Xu_verified[loop_index]:
                    t = e2p(X[:, l_corresp_X2u[0]])[:, loop_index]
                    # TODO: rewrite loop
                    e = err_reprojection_half(cameras[ic].P,
                                              e2p(X[:, l_corresp_X2u[0]])[:,
                                              loop_index],
                                              cameras[ic].interest_points_p[:,
                                              l_corresp_X2u[1]][:, loop_index])
                    if e < THETA:
                        corr_ok.append(loop_index)
            c.verify_x(ic, corr_ok)
        c.finalize_camera()

    # TASK 4 REALLY BEGUN HERE
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    task = []
    Hs = []
    imgs = []
    # if not os.path.exists('stereo_out.mat'):
    for pair in pairs:
        c1 = cameras[pair[0]]
        c2 = cameras[pair[1]]
        l_F = PP2F(c1.P, c2.P)

        Ha, Hb, imga_r, imgb_r = rectify(l_F, c1.img, c2.img)

        Hs.append((Ha, Hb))
        imgs.append((imga_r, imgb_r))

        mX1, mu1, _ = c.get_Xu(pair[0])
        mX2, mu2, _ = c.get_Xu(pair[1])

        restored_idxs = restore_idxs(mX1, mX2, mu1, mu2).T
        points_H_c1 = Ha @ c1.interest_points_p[:, restored_idxs[0]]
        points_H_c2 = Hb @ c2.interest_points_p[:, restored_idxs[1]]
        points_H_c1 /= points_H_c1[-1]
        points_H_c2 /= points_H_c2[-1]

        u_a_r = points_H_c1
        u_b_r = points_H_c2

        seeds = np.vstack(
                (u_a_r[0, :], u_b_r[0], (u_a_r[1] + u_b_r[1]) / 2)).T
        task_i = np.array([imga_r, imgb_r, seeds], dtype=object)
        task += [task_i]
    X = np.array([[], [], []])
    task = np.vstack(task)

    # os.chdir("task_general/gcs/")
    # os.system("pwd")

    scipy.io.savemat('stereo_in.mat', {'task': task})
    print("matlab functions running...")
    os.system('matlab -nodisplay -nosplash -nodesktop -r "run(\'test_gcs.m\');exit;"')
    flab.test_gcs(nargout=0)
    print("stereo matching...")

    d = scipy.io.loadmat('stereo_out.mat')
    D = d['D']
    for i in range(len(pairs)):
        # (x, y) of ima --  (x+Di[y,x],y) of imb
        Di = D[i, 0]
        im_a_r = imgs[i][0]
        im_b_r = imgs[i][1]

        Ha_inv = np.linalg.inv(Hs[i][0])
        Hb_inv = np.linalg.inv(Hs[i][1])
        pixs1_real, pixs2_real = [], []
        for x in range(min(im_a_r.shape[1], im_b_r.shape[1])):
            for y in range(min(im_a_r.shape[0], im_b_r.shape[0])):
                if np.isnan(Di[y][x]):
                    continue
                pix1_real = Ha_inv @ np.array([x, y, 1])
                pix2_real = Hb_inv @ np.array([x + int(Di[y][x]), y, 1])
                pixs1_real.append(pix1_real)
                pixs2_real.append(pix2_real)

        pixs1_real = e2p(p2e(np.array(pixs1_real).T))
        pixs2_real = e2p(p2e(np.array(pixs2_real).T))

        F = PP2F(cameras[pairs[i][0]].P, cameras[pairs[i][1]].P)

        pixs1_real, pixs2_real = u_correct_sampson(F, pixs1_real, pixs2_real)

        tmp = triangulate_3dv(cameras[pairs[i][0]].P,
                              cameras[pairs[i][1]].P,
                              pixs1_real,
                              pixs2_real)

        X = np.append(X, p2e(tmp), axis=1)
        print("next {}".format(i))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    scale = 6
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.set_zlim(-scale, scale)

    origin = np.eye(3)
    d = np.array([0, 0, 0])
    plot_csystem(ax, origin, d, '0', "black")
    array_C, array_t = [], []

    idx = X.T[:, -1].argsort()[::-1]
    num_of_outliers = int(X.shape[1] * 2.5 / 100)
    X = X.T[idx].T[:, num_of_outliers:-num_of_outliers]

    # plot cameras
    # for i in range(NUM_OF_IMGS):
    #     R = cameras[imgs_order[i]].R
    #     t = cameras[imgs_order[i]].t
    #     plot_csystem(ax, R.T, R.T @ -t, 'c{}'.format(imgs_order[i]))
    #     array_C.append(R.T)
    #     array_t.append(R.T @ -t)
    # plot_cameras(ax, array_C, array_t, imgs_order)
    # plot points
    ax.plot3D(X[0], X[1], X[2], 'b,', )
    # ax.plot3D(new_Xs[0], new_Xs[1], new_Xs[2], 'g.')
    ax.plot3D(0, 0, 0, "r.")
    plt.show()

    #  make a point cloud

    g = ge.GePly('out.ply')
    g.points(
            X)  # Xall contains euclidean points (3xn matrix), ColorAll RGB colors (3xn or 3x1, optional)
    for i in range(NUM_OF_IMGS):
        # g.points(np.array([0, 0, 0]).reshape(3, 1), color=np.array([255.0, .0, .0]).reshape(3, 1))
        R = cameras[imgs_order[i]].R
        t = cameras[imgs_order[i]].t
        g.points((R.T @ -t).reshape(3, 1),
                 color=np.array([255.0, .0, .0]).reshape(3, 1))
    g.close()
