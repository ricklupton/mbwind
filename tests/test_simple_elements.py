from numpy import zeros, array, eye, pi, dot, sqrt, c_, diag
from numpy.testing import assert_array_equal, assert_array_almost_equal
from unittest import TestCase

from mbwind.utils import rotmat_x, rotmat_z
from mbwind.elements import Hinge, RigidBody


class HingeTestCase(TestCase):
    def test_transform(self):
        h = Hinge('hinge', [0, 0, 1])
        h.rp = array([3.5, 9.21, 8.6])
        h.Rp = eye(3)

        # Test distal transform
        h.calc_distal_pos()
        assert_array_equal(h.rd, h.rp)
        assert_array_equal(h.Rd, h.Rp)

        h.xstrain[0] = pi / 2
        h.calc_distal_pos()
        assert_array_equal(h.rd, h.rp)  # always coincident
        # New unit vectors X -> y, Y -> -x, Z -> z
        assert_array_almost_equal(h.Rd, c_[[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

        # TODO should test when Rp != I

    def test_post_transform(self):
        h = Hinge('hinge', [0, 0, 1], post_transform=rotmat_x(pi / 2))
        h.rp = array([3.5, 9.21, 8.6])
        h.Rp = eye(3)

        # Test distal transform with post transform
        h.calc_distal_pos()
        assert_array_equal(h.rd, h.rp)
        # just post transform
        assert_array_almost_equal(h.Rd, c_[[1, 0, 0], [0, 0, 1], [0, -1, 0]])

        h.xstrain[0] = pi / 2
        h.calc_distal_pos()
        assert_array_equal(h.rd, h.rp)  # always coincident
        # rotate about z then x
        assert_array_almost_equal(h.Rd, c_[[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    def test_velocity(self):
        h = Hinge('hinge', [0, 0, 1])
        h.vstrain[0] = 4.5  # rad/s
        h.calc_kinematics()

        assert_array_equal(h.F_vp, eye(6))  # distal node moves with prox
        assert_array_equal(h.F_ve[:3, 0], zeros(3))   # hinge ang vel is
        assert_array_equal(h.F_ve[3:, 0], [0, 0, 1])  # about axis
        assert_array_equal(h.F_v2, zeros(6))  # no quadratic force, base fixed

        # TODO should test when Rp != I


class RigidBodyTestCase(TestCase):
    def test_mass_simple(self):
        # simple rigid body at origin
        II = diag([4.2, 6.7, 11.7])
        b = RigidBody('body', mass=4.3, inertia=II)
        b.calc_mass()
        assert_array_equal(b.mass_vv[:3, :3], 4.3 * eye(3))
        assert_array_equal(b.mass_vv[:3, 3:], 0)
        assert_array_equal(b.mass_vv[3:, :3], 0)
        assert_array_equal(b.mass_vv[3:, 3:], II)
        assert_array_equal(b.quad_forces, 0)

        # now rotate node by 90 deg about z axis, swapping x & y
        b.Rp = rotmat_z(pi / 2)
        b.calc_mass()
        assert_array_equal(b.mass_vv[:3, :3], 4.3 * eye(3))
        assert_array_equal(b.mass_vv[:3, 3:], 0)
        assert_array_equal(b.mass_vv[3:, :3], 0)
        assert_array_almost_equal(b.mass_vv[3:, 3:], diag([6.7, 4.2, 11.7]))
        assert_array_equal(b.quad_forces, 0)
