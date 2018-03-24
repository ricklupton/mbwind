import unittest
import numpy as np
from numpy import array, eye, pi, c_, cos, sin, dot
from numpy.testing import assert_array_equal, assert_array_almost_equal

from mbwind import System, RigidBody
from mbwind.elements import FreeJoint
from .common import (random_rotation_matrix, set_random_state,
                     constraint_consistency_check)


def _make_random_joint(rdm):
    # Make free joint with random axis and transform
    return FreeJoint('joint', post_transform=random_rotation_matrix(rdm))


class TestFreeJoint(unittest.TestCase):
    def setUp(self):
        self.rdm = np.random.RandomState(123456789)

    def test_distal_node_position(self):
        NTESTS = 50
        for i in range(NTESTS):
            j = _make_random_joint(self.rdm)
            set_random_state(j, self.rdm, 1)

            # Set simple rotation parameters to check basic rotations
            j.xstrain[3:6] = [pi/3, 0, 0]
            Rj = c_[[1, 0, 0],
                    [0, cos(pi/3), sin(pi/3)],
                    [0, -sin(pi/3), cos(pi/3)]]
            j.calc_distal_pos()
            assert_array_almost_equal(j.rd,
                                      j.rp + dot(j.Rp, j.xstrain[0:3]))
            assert_array_almost_equal(j.Rd,
                                      dot(j.Rp, dot(Rj, j.post_transform)))

            # Check combination of Euler angles: 90deg yaw and pitch
            j.xstrain[3:6] = [0, pi/2, pi/2]
            Rj = c_[[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
            j.calc_distal_pos()
            assert_array_almost_equal(j.rd,
                                      j.rp + dot(j.Rp, j.xstrain[0:3]))
            assert_array_almost_equal(j.Rd,
                                      dot(j.Rp, dot(Rj, j.post_transform)))

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

    def test_velocity_and_acceleration_constraints_are_consistent(self):
        # Check for consistency with random conditions
        constraint_consistency_check(_make_random_joint, self.rdm)

    def test_interal_forces(self):
        # NB minus sign because convention for applied_stress is that
        # stiffness loads are positive.
        j = FreeJoint('joint')

        # Constant loading
        j.internal_forces = np.array([2, 3.1, 2.1, 4.3, 2.5, 1.0])
        j.calc_external_loading()
        assert_array_equal(j.applied_forces, 0)
        assert_array_equal(j.applied_stress, -j.internal_forces)

        # Loading function
        j.internal_forces = lambda element, t: np.ones(6)
        j.calc_external_loading()
        assert_array_equal(j.applied_forces, 0)
        assert_array_equal(j.applied_stress, -1)

    def test_distal_forces(self):
        j = FreeJoint('joint')

        # Constant loading
        j.distal_forces = np.array([2, 3.1, 2.1, 4.3, 2.5, 1.0])
        j.calc_external_loading()
        assert_array_equal(j.applied_forces[:6], 0)
        assert_array_equal(j.applied_forces[6:], j.distal_forces)
        assert_array_equal(j.applied_stress, 0)

        # Loading function
        j.distal_forces = lambda element, t: np.ones(6)
        j.calc_external_loading()
        assert_array_equal(j.applied_forces[:6], 0)
        assert_array_equal(j.applied_forces[6:], 1)
        assert_array_equal(j.applied_stress, 0)

    def test_distal_forces_cause_acceleration(self):
        j = FreeJoint('joint')
        b = RigidBody('body', mass=3, inertia=np.diag([7, 7, 7]))
        s = System()
        s.add_leaf(j)
        j.add_leaf(b)
        s.setup()

        # Constant loading
        j.distal_forces = np.array([2, 0, 0, 0, 0, 0])
        s.update_kinematics()
        s.update_matrices()
        s.solve_accelerations()
        s.update_kinematics()
        assert_array_equal(j.ad, [2. / 3, 0, 0, 0, 0, 0])
