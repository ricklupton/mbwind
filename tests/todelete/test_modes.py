from numpy import (zeros, array, eye, pi, dot, sqrt, c_, diag, cos, sin,
                   asarray, arange, ones_like, zeros_like, trapz)
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal as assert_aae)

from mbwind.modes import ModesFromScratch, ModalRepresentation
from mbwind.utils import rotmat_z
from nose.tools import *

class ModesFromScratch_Tests:
    def test_transform(self):
        """
        Test transformation matrix from global to local coordinates
        """

        # values not important for this test
        modes = ModesFromScratch(arange(10), 1, 1, 1, 1)

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
        assert_aae(  # noqa
            dot(Y, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
                   [0, 0, 0, 0, 0, 0, c, -s, 0, 0, 0, 0])
        assert_aae(  # noqa
            dot(Y, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
                   [0, 0, 0, 0, 0, 0, c, s, 0, 0, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        assert_aae(  # noqa
            dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 0, c, s, 0]),
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        # Check now what happens if proximal node has a velocity
        assert_aae(  # noqa
            dot(Y, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                   [0, 0, 0, 0, 0, 0, -c, s, 0, 0, 0, 0])
        assert_array_equal(dot(Y, [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


class ModalRepesentationWithNoModes_Tests:
    def setup(self):
        self.x = arange(0, 10.1, 1)
        self.L = self.x[-1]
        self.modes = ModalRepresentation(self.x)

    def _uniform_force(self, direction, magnitude):
        P = zeros((len(self.x), 3))
        P[:, direction] = magnitude
        return P

    def test_distributed_loading_for_uniform_axial_loading(self):
        P = self._uniform_force(0, 3.4)
        Qr, Qw, Qe = self.modes.distributed_loading(P, [])
        assert_aae(Qr, [3.4 * self.L, 0, 0])
        assert_aae(Qw, [0, 0, 0])         # axial so no moment
        assert_equal(Qe.shape, (0,))

    def test_distributed_loading_for_uniform_perpendicular_loading(self):
        P = self._uniform_force(1, 3.4)
        Qr, Qw, Qe = self.modes.distributed_loading(P, [])
        assert_aae(Qr, [0, 3.4 * self.L, 0])
        assert_aae(Qw, [0, 0, 3.4 * self.L**2 / 2])
        assert_equal(Qe.shape, (0,))

class ModalRepesentationWithNoModes_Tests:
    def setup(self):
        # Define a mode shape with deflection x^2 in the y direction
        self.x = x = arange(0, 10.1, 1)
        self.L = self.x[-1]
        zero = 0*x
        shapes    = c_[zero, x**2, zero].reshape((len(x), 3, 1))
        rotations = c_[zero, 2*x,  zero].reshape((len(x), 3, 1))
        self.modes = ModalRepresentation(x, freqs=[1], shapes=shapes,
                                         rotations=rotations)

    def _uniform_force(self, direction, magnitude):
        P = zeros((len(self.x), 3))
        P[:, direction] = magnitude
        return P

    def test_distributed_loading_for_uniform_axial_loading(self):
        P = self._uniform_force(0, 3.4)
        Qr, Qw, Qe = self.modes.distributed_loading(P, [0])
        assert_aae(Qr, [3.4 * 10, 0, 0])
        assert_aae(Qw, [0, 0, 0])         # axial so no moment
        assert_aae(Qe, [0])               # axial so no strain force

    def test_distributed_loading_uniform_loading_aligned_with_mode(self):
        P = self._uniform_force(1, 3.4)
        Qr, Qw, Qe = self.modes.distributed_loading(P, [0])
        assert_aae(Qr, [0, 3.4 * 10, 0])
        assert_aae(Qw, [0, 0, 3.4 * self.L**2 / 2])
        # Generalised force should be integral of applied
        # force and deflection. Integral of x^2 if L^3 / 3
        assert_aae(Qe, [3.4 * trapz(self.x**2, self.x)])

    def test_distributed_loading_uniform_loading_not_aligned_with_mode(self):
        P = self._uniform_force(2, 3.4)
        Qr, Qw, Qe = self.modes.distributed_loading(P, [0])
        assert_aae(Qr, [0, 0, 3.4 * 10])
        assert_aae(Qw, [0, -3.4 * self.L**2 / 2, 0])
        assert_aae(Qe, [0])
