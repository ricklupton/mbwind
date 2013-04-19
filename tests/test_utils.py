from mbwind.utils import (skewmat, update_skewmat, rotmat_x, rotmat_y,
                          rotmat_z, discont_trapz)
from numpy import ones, zeros, array, arange, eye, pi, dot, sqrt
from numpy.testing import assert_array_equal, assert_array_almost_equal
from unittest import TestCase


class UtilsTestCase(TestCase):
    def test_skew(self):
        assert_array_equal(skewmat(zeros(3)), zeros((3, 3)))
        assert_array_equal(skewmat([1, 2, 3]), array([
            [0, -3,  2],
            [3,  0, -1],
            [-2, 1,  0],
        ]))

    def test_update_skew(self):
        x = zeros((3, 3))
        update_skewmat(x, [0, 0, 0])
        assert_array_equal(x, zeros((3, 3)))
        update_skewmat(x, [1, 2, 3])
        assert_array_equal(x, array([
            [0, -3,  2],
            [3,  0, -1],
            [-2, 1,  0],
        ]))

    def test_rotmats(self):
        assert_array_equal(rotmat_x(0), eye(3))
        assert_array_equal(rotmat_y(0), eye(3))
        assert_array_equal(rotmat_z(0), eye(3))

        # Rotate 45 deg about each axis
        assert_array_almost_equal(dot(rotmat_x(pi / 4), [1, 1, 1]),
                                  [1, 0, sqrt(2)])
        assert_array_almost_equal(dot(rotmat_y(pi / 4), [1, 1, 1]),
                                  [sqrt(2), 1, 0])
        assert_array_almost_equal(dot(rotmat_z(pi / 4), [1, 1, 1]),
                                  [0, sqrt(2), 1])

    def test_discont_trapz(self):
        x = arange(0, 2.6, 0.5)

        # Integrate constant value with constant weighting
        y = ones((6, 7))
        rho = ones((5, 2))
        assert_array_equal(discont_trapz(y, rho, x), ones(7) * x[5])

        # Integrate constant value over step function weighting
        y = ones((6, 7))
        rho = zeros((5, 2))
        rho[:2] = 1
        assert_array_equal(discont_trapz(y, rho, x), ones(7) * x[2])

        # Integrate linear function between x2 and x4
        y = x
        rho = zeros((5, 2))
        rho[2:4] = 1
        assert_array_equal(discont_trapz(y, rho, x), (x[4]**2 - x[2]**2) / 2)
