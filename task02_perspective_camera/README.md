# Task 0-2: Perspective camera
Develop a simulation of perspective camera projection – wire-frame model. Let the 3D object be given. The object is composed from two planar diagrams, that are connected. Coordinates of vertices of both diagrams X1 and X2 are:
```
X1 = [ [-0.5,  0.5, 0.5, -0.5, -0.5, -0.3, -0.3, -0.2, -0.2,  0,  0.5 ],
       [-0.5, -0.5, 0.5,  0.5, -0.5, -0.7, -0.9, -0.9, -0.8, -1, -0.5 ],
       [ 4,    4,   4,    4,    4,    4,    4,    4,    4,    4,  4   ] ]

X2 = [ [-0.5,  0.5, 0.5, -0.5, -0.5, -0.3, -0.3, -0.2, -0.2,  0,    0.5 ],
       [-0.5, -0.5, 0.5,  0.5, -0.5, -0.7, -0.9, -0.9, -0.8, -1,   -0.5 ],
       [ 4.5,  4.5, 4.5,  4.5,  4.5,  4.5,  4.5,  4.5,  4.5,  4.5,  4.5 ] ]

```
Wire-frame model contains edges, that connects vertices in X1 and X2 in given order, and additionally it contains edges connecting vertices between X1 and X2, such that the vertex X1(:,i) is connected to the vertex X2(:,i), ∀ i.

The internal calibration matrix of the camera is:
```
K = [[ 1000,    0, 500 ],
     [    0, 1000, 500 ],
     [    0,    0,   1 ]]
         
```
### Steps
- Construct following camera matrices (keep the image u-axis parallel to the scene x-axis):
    - P1: camera in the origin looking in the direction of z-axis.
    - P2: camera located at [0,-1,0] looking in the direction of z-axis.
    - P3: camera located at [0,0.5,0] looking in the direction of z-axis.
    - P4: camera located at [0,-3,0.5], with optical axis rotated by 0.5 rad around x-axis towards y-axis.
    - P5: camera located at [0,-5,4.2] looking in the direction of y-axis.
    - P6: camera located at [-1.5,-3,1.5], with optical axis rotated by 0.5 rad around y-axis towards x-axis (i.e., -0.5 rad) followed by a rotation by 0.8 rad around x-axis towards y-axis.
- Use the cameras P1 to P6 for projection of given wire-frame model into an image. The edges inside X1 should be drawn red, the edges inside X2 should be drawn blue and the rest should be drawn in black.

### Hints

Let u1, v1 be the image coordinates of projected vertices X1 and u2, v2 be the coordinates of projected vertices X2. The desired picture can be drawn, e.g., in python like self:
```
plt.plot( u1, v1, 'r-', linewidth=2 )
plt.plot( u2, v2, 'b-', linewidth=2 )
plt.plot( [u1, u2], [v1, v2 ], 'k-', linewidth=2 )
plt.gca().invert_yaxis()
plt.axis( 'equal' ) # self kind of plots should be isotropic
```