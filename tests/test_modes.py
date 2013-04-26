from numpy import zeros, array, eye, pi, dot, sqrt, c_, diag, cos, sin, asarray, arange
from numpy.testing import assert_array_equal, assert_array_almost_equal
from unittest import TestCase

from mbwind.modes import ModesFromScratch
from mbwind.utils import rotmat_z


class ModesFromScratchTestCase(TestCase):
    def test_transform(self):
        """
        Test transformation matrix from global to local coordinates
        """

        # values not important for this test
        modes = ModesFromScratch(arange(10), 1, 1, 1)

        def getY(Rp, rp, rd):
            rp, rd = asarray(rp), asarray(rd)
            return modes.transformation_to_global_coords(Rp, rp, rd)

        ###### Simple case with Rp = I ######
        Y = getY(eye(3), [0, 0, 0], [1, 0, 0])

        # Velocity of distal node directly becomes displacement if prox fixed
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        # Check now what happens if proximal node has a velocity
        assert_array_equal(dot(Y, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0])
        assert_array_equal(dot(Y, [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1])

        ###### Now with Rp = rotmat_z(45 deg) ######
        Y = getY(rotmat_z(pi/4), [0, 0, 0], [1, 0, 0])
        c, s = cos(pi/4), sin(pi/4)

        # Input velocities aligned with global axes are transformed
        assert_array_almost_equal(  # noqa
            dot(Y, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
                   [0, 0, 0, 0, 0, 0, c, -s, 0, 0, 0, 0])
        assert_array_almost_equal(  # noqa
            dot(Y, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
                   [0, 0, 0, 0, 0, 0, c, s, 0, 0, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        assert_array_almost_equal(  # noqa
            dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 0, c, s, 0]),
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        # Check now what happens if proximal node has a velocity
        assert_array_almost_equal(  # noqa
            dot(Y, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                   [0, 0, 0, 0, 0, 0, -c, s, 0, 0, 0, 0])
        assert_array_equal(dot(Y, [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
