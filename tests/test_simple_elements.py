from numpy import zeros, array, eye, pi, dot, sqrt, c_, diag, cos, sin
from numpy.testing import assert_array_equal, assert_array_almost_equal
from unittest import TestCase

from mbwind.utils import rotmat_x, rotmat_z
from mbwind.elements import Hinge, FreeJoint, RigidBody


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


class FreeJointTestCase(TestCase):
    def test_transform(self):
        j = FreeJoint('joint')
        j.rp = array([3.5, 9.21, 8.6])
        j.Rp = eye(3)

        # Test distal transform -- translation
        j.xstrain[:3] = [2, 3, 4]
        j.calc_distal_pos()
        assert_array_equal(j.rd, j.rp + [2, 3, 4])
        assert_array_equal(j.Rd, j.Rp)

        # Now add a rotation of 60 deg about x axis
        j.xstrain[3] = pi / 3
        j.calc_distal_pos()
        assert_array_equal(j.rd, j.rp + [2, 3, 4])
        # new unit vectors after rotation
        assert_array_almost_equal(j.Rd, c_[[1, 0, 0],
                                           [0, cos(pi/3), sin(pi/3)],
                                           [0, -sin(pi/3), cos(pi/3)]])

        # Check combination of Euler angles: 90deg yaw and pitch
        j.xstrain[3:] = [0, pi/2, pi/2]
        j.calc_distal_pos()
        assert_array_equal(j.rd, j.rp + [2, 3, 4])
        # new unit vectors after rotation.
        #  1) 90deg yaw    ->   y, -x, z
        #  2) 90deg pitch  ->  -z, -x, y
        assert_array_almost_equal(j.Rd, c_[[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

    def test_velocity(self):
        j = FreeJoint('joint')
        j.rp = array([0, 0, 8.6])

        # x-velocity: shouldn't depend on speed
        j.vstrain[:] = [2.3, 0, 0, 0, 0, 0]
        j.calc_kinematics()
        assert_array_equal(j.F_vp, eye(6))  # movement of p node moves d node
        assert_array_equal(j.F_ve, eye(6))  # with zero rotation, direct
        assert_array_equal(j.F_v2, 0)

        # now apply a 90 deg yaw angle (about z axis): now 'pitch'
        # causes a -ve global x-rotation, and 'roll' causes a +ve
        # global y-rotation
        j.xstrain[5] = pi / 2
        j.calc_kinematics()
        assert_array_equal(j.F_vp, eye(6))  # movement of p node moves d node
        assert_array_equal(j.F_v2, 0)
        F_ve = eye(6)
        F_ve[3:, 3] = [0,  1, 0]
        F_ve[3:, 4] = [-1, 0, 0]
        assert_array_almost_equal(j.F_ve, F_ve)

        # TODO test quadratic velocity vector


class RigidBodyTestCase(TestCase):
    def test_mass_simple(self):
        # simple rigid body at origin: this just checks the mass and
        # inertia carry directly through to element mass matrix
        II = diag([4.2, 6.7, 11.7])
        b = RigidBody('body', mass=4.3, inertia=II)
        b.calc_mass()
        assert_array_equal(b.mass_vv[:3, :3], 4.3 * eye(3))
        assert_array_equal(b.mass_vv[:3, 3:], 0)
        assert_array_equal(b.mass_vv[3:, :3], 0)
        assert_array_equal(b.mass_vv[3:, 3:], II)
        assert_array_equal(b.quad_forces, 0)

        # now rotate node by 90 deg about z axis, swapping x & y. This
        # confirms that the mass part doesn't change, but the inertia
        # matrix is updated.
        b.Rp = rotmat_z(pi / 2)
        b.calc_mass()
        assert_array_equal(b.mass_vv[:3, :3], 4.3 * eye(3))
        assert_array_equal(b.mass_vv[:3, 3:], 0)
        assert_array_equal(b.mass_vv[3:, :3], 0)
        assert_array_almost_equal(b.mass_vv[3:, 3:], diag([6.7, 4.2, 11.7]))
        assert_array_equal(b.quad_forces, 0)

    def test_mass_offset(self):
        # check mass matrix calculation when centre of mass is offset
        # from proximal node.
        b = RigidBody('body', mass=5.6, Xc=[1.2, 3.4, 5.4])
        b.calc_mass()

        # F = ma: a Z acceleration should produce a Z-force and X- and
        # Y-moments.
        F = -dot(b.mass_vv, [0, 0, 1, 0, 0, 0])
        assert_array_equal(F, b.mass * array([0, 0, -1, -3.4, 1.2, 0]))

    def test_weight_offset(self):
        # check applied force due to gravity is correct
        b = RigidBody('body', mass=5.6, Xc=[1.2, 3.4, 5.4])
        b.calc_mass()
        b.calc_external_loading()
        assert_array_equal(b.applied_forces,
                           b.mass * 9.81 * array([0, 0, -1, -3.4, 1.2, 0]))
