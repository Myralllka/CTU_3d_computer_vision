# ge - Package for export of 3D geometry
#
# (c) 2020-11-23 Martin Matousek
# Last change: $Date$
#              $Revision$

import numpy as np


class Ge:
    """ 3D geometry export - base class defining the interface. """

    def close(self):
        raise Exception("Unimplemented.")

    def points(self):
        raise Exception("Unimplemented.")

    @staticmethod
    def points_arg_helper(X, color=None, colortype=None):

        npt = X.shape[1]

        if color is not None:
            if isinstance(color, list) or isinstance(color, tuple):
                color = np.array([color]).T

            nc = color.shape[1]

            if (len(color.shape) != 2 or color.shape[0] != 3 or
                    (nc != npt and nc != 1)):
                raise Exception("The color must be 3x1 or 3xn array.")

            if nc == 1:
                color = np.tile(color, (1, npt))

            if color.dtype != 'float' and color.dtype != 'uint8':
                raise Exception("Unhandled data type of the color(s).")

            if colortype is None:
                pass  # no conversion, keep as is

            elif colortype == 'uint8':
                if color.dtype == 'float':
                    color = (color * 255.0).astype('uint8')

            elif colortype == 'double':
                if color.dtype == 'uint8':
                    color = (color * 255.0).astype('uint8')
            else:
                raise Exception("Unhandled value for colortype.")

        bad = np.isnan(X).any(axis=0)

        if bad.any():
            ok = ~bad
            X = X[:, ok]

            if color is not None:
                color = color[:, ok]

        return X, color


class GePly(Ge):
    """ 3D geometry export into PLY file. """

    def __init__(self, file, fmt='binary'):
        """
        Constructor.

        obj = GePly( file, fmt='binary' )

          fmt - 'binary' or 'ascii'
        """

        if fmt != 'binary' and fmt != 'ascii':
            raise Exception('Unknown ply format requested.')

        self.fh = open(file, 'wt')  # opened ply file handle
        self.binary = fmt == 'binary'  # PLY format: true = binary, false = ascii

        self.vertices = []  # list of subarrays of vertices
        self.colors = []  # list of subarrays of corresponding vertex colours
        self.vcount = 0  # total count of vertices

        if self.binary:
            print('ply\nformat binary_little_endian 1.0\n',
                  file=self.fh, end='')
        else:
            print('ply\nformat ascii 1.0\n', file=self.fh, end='')

    def close(self):
        """
        Finish and close the PLY file.

        obj.close()
        """

        if self.fh is None:
            return

        is_colors = False
        for c in self.colors:
            if c is not None:
                is_colors = True

        if is_colors:
            for ci in range(len(self.colors)):
                if self.colors[ci] is None:
                    c = np.ones(np.shape(self.vertices[ci])) * 255
                    self.colors[ci] = c.astype('uint8')

        # write head (colours are used only if needed)
        print('element vertex', self.vcount, file=self.fh)
        print('property float x', file=self.fh)
        print('property float y', file=self.fh)
        print('property float z', file=self.fh)
        if is_colors:
            print('property uchar red', file=self.fh)
            print('property uchar green', file=self.fh)
            print('property uchar blue', file=self.fh)

        print('end_header', file=self.fh)

        # write data
        if self.binary:
            # vertices
            for i in range(len(self.vertices)):
                v = self.vertices[i].astype('float32').view('uint8')
                if is_colors:
                    c = self.colors[i].view('uint8')
                    v = np.hstack((v, c))
                v.tofile(self.fh)

        else:
            # vertices
            for i in range(len(self.vertices)):
                v = self.vertices[i]
                if is_colors:
                    c = self.colors[i]
                    v = np.hstack((v, c))
                    np.savetxt(self.fh, v, '%f %f %f %i %i %i')
                else:
                    np.savetxt(self.fh, v, '%f %f %f')

        self.fh.close()
        self.fh = None

    def points(self, X, color=None):
        """
        Export of points.

        obj.points( X, color=None )

        color: None or 3x1 or 3xN numpy array of rgb values
        """

        X, color = self.points_arg_helper(X, color, colortype='uint8')

        self.vcount += X.shape[1]

        self.vertices += [np.ndarray.copy(X.T, order='C')]

        if color is None:
            self.colors += [color]
        else:
            self.colors += [np.ndarray.copy(color.T, order='C')]
