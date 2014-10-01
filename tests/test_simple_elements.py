import unittest
import random
import numpy
from numpy import zeros, array, eye, pi, dot, c_, diag, cos, sin
from numpy.testing import assert_array_equal, assert_array_almost_equal

from mbwind.utils import rotmat_x, rotmat_y, rotmat_z, update_skewmat
from mbwind import System
from mbwind.elements import Hinge, FreeJoint, RigidBody


class TestHinge(unittest.TestCase):
    def test_distal_node_is_rotated_by_90deg_about_correct_axis(self):
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

    def test_additional_post_transform_of_90deg_is_applied(self):
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

    def test_distal_node_velocity_due_to_hinge_rotation_has_correct_axis(self):
        h = Hinge('hinge', [0, 0, 1])
        h.vstrain[0] = 4.5  # rad/s
        h.calc_kinematics()

        assert_array_equal(h.F_vp, eye(6))  # distal node moves with prox
        assert_array_equal(h.F_ve[:3, 0], zeros(3))   # hinge ang vel is
        assert_array_equal(h.F_ve[3:, 0], [0, 0, 1])  # about axis
        assert_array_equal(h.F_v2, zeros(6))  # no quadratic force, base fixed

        # TODO should test when Rp != I


class TestFreeJoint(unittest.TestCase):
    def test_distal_node_is_transformed_by_joint_freedoms(self):
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

    def test_velocity_transforms_depend_on_joint_orientation(self):
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

    def test_velocity_and_acceleration_are_consistent(self):
        NTESTS = 100
        DT = 1e-3
        RANGE = 1
        EXPECTED_DECIMAL = -int(numpy.log10(RANGE*DT))
        rdm = random.Random(123456789)

        def set_random_values(arr):
            arr[:] = [rdm.uniform(-RANGE, RANGE) for j in range(len(arr))]
        j = FreeJoint('joint')
        for i in range(NTESTS):
            print(i)
            # Calculate angular velocity and angular acceleration for
            # random conditions
            set_random_values(j.xstrain)
            set_random_values(j.vstrain)
            print(j.xstrain)
            print(j.vstrain)
            j.update_kinematics()  # NB need update_kinematics() to
                                   # update vd & ad, not
                                   # calc_kinematics()
            vd1 = j.vd.copy()
            ad1 = j.ad.copy()

            # Peturb by small timestep & recalculate
            j.xstrain[:] += j.vstrain[:] * DT
            j.update_kinematics()
            vd2 = j.vd.copy()

            assert_array_almost_equal(ad1, (vd2 - vd1) / DT, EXPECTED_DECIMAL)


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
