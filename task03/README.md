# Task 0-3: Robust Maximum Likelihood Estimation of a Line(s) From Points

A 2D line [-10, 3, 1200] generates set of 100 points (inliers), corrupted by a gaussian noise. Additionally, there is a set of 200 uniformly distributed random points not belonging to the line (outliers). There are three sets of points linefit_1.txt, linefit_2.txt, linefit_3.txt generated with the gaussian noise σ=1, σ=2, and σ=3, respectively. 

For such kind of data, non-robust estimation method, e.g. regression by least squares, cannot be used. A golden standard method for robust (i.e. in the presence of outliers) estimation in such a case is RANSAC. If more acurate result is needed, additional optimisation is applied to the result of RANSAC, i.e. least squares regression using the inliers only. One of the common modifications of RANSAC method employs support function derived as a maximum likelihood estimate (MLE), instead of a standard zero-one box function used in traditional RANSAC. This modification (MLESAC) has the same complexity and convergence, but tends to produce more accurate estimates directly. 

To show difference between results of the above methods, we run RANSAC and MLESAC on a single data set 100 times. Every run is then followed by least squares line regression using inliers. For visualisation purposes, every line is characterised by two parameters - its angle α and its orthogonal distance d to origin, thus showing as a single point in α−d graph.

### Steps

- Choose at least one set of points from linefit_1.txt, linefit_2.txt, linefit_3.txt.
- Use these points for (non-robust) least squares estimation (regression) of the line. This is expected to fail.
- Use these points for robust estimation of the line. Use RANSAC to find initial estimate and to separate inliers and outliers, followed by least squares regression using the inliers.
- Repeat the previous step with MLESAC.
- For the estimation, do not use the parameters of the original generator (numbers of inliers, outliers, σ). Set a RANSAC threshold by hand.
- Draw the whole situation:
    - Original line.
    - Points.
    - Line found by non-robust regression.
    - Line found by RANSAC
    - Line found by RANSAC followed by least squares regression
    - Line found by MLESAC followed by least squares regression

### Hints
Reading a point coordinates from a text file can be achieved as x = load( 'linefit_1.txt' )'; (matlab) or x=numpy.loadtxt( 'linefit_1.txt' ).T (python); note the transpose to obtain column vectors. Random sample of 2 from n points can be generated as i = randperm( n, 2 ); or rng = numpy.random.default_rng() called once followed by i = rng.choice( n, 2, replace=False ).