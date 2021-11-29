# Input Data
## Images of the Scene

The goal of the work during the whole term is to reconstruct a 3D object (scene) from its images. In order to make the task manageable, we have chosen such a scene, that is relatively uncomplicated considering 3D computer vision methods: a decorative portal. The simplicity of such a scene lies in the fact, that the scene is almost planar and small number of views is enough for reconstruction.<br>

We have prepared data captured at several places, see `imgs`. The capturing scheme consists of three levels of height, four pictures in each. The capturing scheme is in figure 1. All images are captured with the same zoom setting, i.e. the same internal camera calibration matrix K.<br> 

## Sparse Correspondences
The sparse correspondences for provided scenes has been computed and they are available. Note, that the correspondences are tentative, so they contain also mismatches.<br>

The correspondences are stored in several files (`data`):<br>
- Coordinates of detected points are stored in the files u_<id>.txt, one row per coordinate pair x y, where <id> is the image identifier (two digit number)<br>
- 0-based indices of corresponding points are stored in the files m_<i1>_<i2>.txt, one row per pair of indices, where <i1> and <i2> are identifiers of two images, i1 < i2<br>
Example:<br>
The file m_05_06.txt begins with,
```
 4 7
 5 18285
11 27631
...
```
the file u_05.txt begins with
```
 5.3 1613.4
 7.0  364.8
 9.5 1522.3
 9.9  585.1
10.9  571.7  <---
11.2  578.6
11.3  666.1
...
```
and the file u_06.txt begins with
```
 6.3 1749.0
 8.4 1753.3
 8.9  497.9
10.4  540.9
11.0  683.2
11.0  687.8
11.1  589.8
11.3  583.4  <---
12.1 1212.6
12.2  949.3
...
```
The first row in m_05_06.txt means, that the point 4 from the image 05 and the point 7 from the image 06 corresponds. I.e., the point with coordinates x=10.9, y=571.7 in the image 05 corresponds to the point x=11.3, y=583.4 in the image 06.

# Robust Estimation of Calibrated Epipolar Geometry of Image Pairs
Calibrated epipolar geometry is characterised by essential matrix, that encodes relative translation and rotation. The matrix can be estimated using image-to-image correspondences. Since the tentative correspondences are assumed to contain errors, a robust estimation method should be used. Then an epipolar geometry is found, together with a sub-set of correspondences that are consistent with it (allowing some defined inaccuracy). One possible method is the RANSAC scheme: hypothesis of essential matrix is generated and its consistence with data (correspondences) is verified. These two steps are repeated and the hypothesis most consistent with data is chosen.
### Hypothesis Generation
The images are captured by a calibrated camera (K is known), so the corresponding image points are first transformed by K^{-1}. Then the essential matrix E is estimated. This matrix has five degrees of freedom and can be computed from five point correspondences. The five-point algorithm is available in code repository (package p5). Then the hypothesis generation can be summarised by following procedure:<br>
- Chose a random 5-tuple of correspondences (points u1 in the first image and u2 in the second one).<br>
- Compute essential matrices E from u1 a u2. There can be more than one solution given by the five-point algorithm typically. Each one must be treated independently.<br>
- Decompose every E into rotations R and translations t - four combinations. The rotation and translation can be used to construct a camera pair and to reconstruct the 3D positions of the five corresponding points. Select such R, t from the four combinations for which the five reconstructed 3D points lie in front of both cameras. For a particular E there can be no solution as well.<br>
In order to test that the 3D points are in front of cameras, the cameras necessary for triangulation of u1 and u2 must be constructed. Since the points have allready K undone, the cameras will be P1 = [I|0] (canonical one) and P2 = [R|t].
### Verification of Consensus of Hypothesis and Data
Every pair of R, t must be verified against all correspondences. Given some robust function, a measure of consensus (a support) is computed as a sum of contributions of all single correspondences. A reprojection eroor (using e.g. Sampson approximation) must be estimated for each correspondence. A contribution of every single correspondence is then computed from self error by chosen robust function. In practice, we can use (figure 1):<br>
- thresholding: if the error is lower than threshold, count 1; otherwise count 0
- approximation of ML estimator: if the error err is lower than threshold thr, count 1-\frac{err^2}{thr^2}, otherwise count 0.
The correspondences with reprojection error below the threshold are assumed correct (inliers), the others are assumed incorrect (outliers). The correspondences with error below the threshold must be again verified if they lead to points in front of both cameras (e.g. by triangulation), and the bad ones must be considered as outliers and its support counted as 0. <br>

As a result of robust estimation, there is epipolar geometry characterised by rotation R and translation t and separation of correspondences into inliers and outliers<br>

#### Task 2
- Implement calibrated estimation of epipolar geometry.
- Chose a single image pair and show inlier and outlier correspondences.
- Select a reasonable sub-set of inliers (every n-th) and show corresponding points and epipolar lines in both images. Use different colours, plot corresponding points and epipolar lines in the same colour.
