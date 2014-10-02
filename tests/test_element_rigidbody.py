import unittest
import numpy as np
from numpy import array, eye, pi, dot, diag
from numpy.testing import assert_array_equal, assert_array_almost_equal

from mbwind.utils import rotmat_y, rotmat_z, update_skewmat
from mbwind import System
from mbwind.elements import RigidBody


class TestRigidBody(unittest.TestCase):
    def test_calculates_mass_and_inertia_in_global_coordinates(self):
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

    def test_accounts_for_offset_centre_of_mass_in_mass_matrix(self):
        # check mass matrix calculation when centre of mass is offset
        # from proximal node.
        b = RigidBody('body', mass=5.6, Xc=[1.2, 3.4, 5.4])
        b.calc_mass()

        # F = ma: a Z acceleration should produce a Z-force and X- and
        # Y-moments.
        F = -dot(b.mass_vv, [0, 0, 1, 0, 0, 0])
        assert_array_equal(F, b.mass * array([0, 0, -1, -3.4, 1.2, 0]))

    def test_accounts_for_offset_centre_of_mass_in_applied_force(self):
        # check applied force due to gravity is correct
        b = RigidBody('body', mass=5.6, Xc=[1.2, 3.4, 5.4])
        s = System(gravity=9.81)  # need a System to define gravity
        s.add_leaf(b)
        s.setup()

        b.calc_mass()
        b.calc_external_loading()
        assert_array_equal(b.applied_forces,
                           b.mass * 9.81 * array([0, 0, -1, -3.4, 1.2, 0]))

    def test_has_gyroscopic_forces_when_spinning(self):
        # When the body is spinning, a torque should cause a
        # perpendicular acceleration

        # Set up rigid body spinning about x axis, and precessing about z axis
        precession = 0.1
        spin = 27.3
        A = 2.4           # perpendicular inertia
        C = 5.7           # polar inertia

        b = RigidBody('body', mass=7.04, inertia=diag([A, A, C]))
        b.Rp[:, :] = rotmat_y(pi/2)
        b.vp[3:] = [spin, 0, precession]
        update_skewmat(b.wps, b.vp[3:])
        b.calc_mass()

        # Expect moment of momentum to be [C*spin, 0, A*precession] in global
        Jp = dot(b.Rp, dot(b.inertia, b.Rp.T))
        hp = dot(Jp, b.vp[3:])
        assert_array_almost_equal(hp, [C*spin, 0, A*precession])

        # Expect the torque to be (precession * spin_speed) * (C - A)
        # about the y axis
        expected_Q2 = spin * precession * (C - A)
        assert_array_almost_equal(b.quad_forces, [0, 0, 0, 0, expected_Q2, 0])

    def test_interal_forces(self):
        # NB minus sign because convention for applied_stress is that
        # stiffness loads are positive.
        b = RigidBody('body', 1)

        # Constant loading
        b.nodal_load = np.array([2, 3.1, 2.1])
        b.calc_external_loading()
        assert_array_equal(b.applied_forces, [2, 3.1, 2.1, 0, 0, 0])
        assert_array_equal(b.applied_stress, [])

        # Loading function
        b.nodal_load = lambda element, t: np.ones(3)
        b.calc_external_loading()
        assert_array_equal(b.applied_forces, [1, 1, 1, 0, 0, 0])
        assert_array_equal(b.applied_stress, [])
