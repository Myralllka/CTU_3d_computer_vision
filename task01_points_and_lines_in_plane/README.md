# Task 0-1: Points and Lines in a Plane
This is a simple task that demonstrates working with homogeneous planar points and lines.
### Steps

- Let the image area has an extent [1, 1] to [800, 600]. Draw its boundary. Use the image coordinate system, i.e., x-axis pointing right and y-axis pointing down.
- In your solution, allow entering two pairs of points within self area and display them.
- Calculate the straight line passing through the first pair and the straight line passing through the second pair. Use homogeneous representation. Display the intersection of each line with the image area.
- Calculate the intersection of both lines and draw it, if it is inside the image area.
- Apply the following homography to all entities (points, lines, image boundary) and draw the result to another figure.

```
 K = [ [ 1,     0.1,   0 ], 
        [ 0.1,   1,     0 ],
        [ 0.004, 0.002, 1 ] ]
```
### Hints
For entering the points, the Matlab function ginput, or the python function matplotlib.pyplot.ginput, is suitable. Cross product of two vectors can be computed using the function cross or numpy.cross.