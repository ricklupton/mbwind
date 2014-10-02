import unittest
import numpy as np
from numpy import zeros, array, eye, pi, c_
from numpy.testing import assert_array_equal, assert_array_almost_equal

from mbwind.utils import rotmat_x
from mbwind.elements import Hinge
from .common import constraint_consistency_check, random_rotation_matrix


# Make hinge with random axis and transform
def _make_random_hinge(rdm):
    return Hinge('hinge',
                 hinge_axis=random_rotation_matrix(rdm)[:, 0],
                 post_transform=random_rotation_matrix(rdm))


class TestHinge(unittest.TestCase):
    def setUp(self):
        self.rdm = np.random.RandomState(123456789)

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

    def test_velocity_and_acceleration_constraints_are_consistent(self):
        # Check for consistency with random conditions
        constraint_consistency_check(_make_random_hinge, self.rdm)

    def test_interal_torque(self):
        # NB minus sign because convention for applied_stress is that
        # stiffness loads are positive.
        h = Hinge('hinge', [0, 0, 1])

        # Constant loading
        h.internal_torque = 3.4
        h.calc_external_loading()
        assert_array_equal(h.applied_forces, 0)
        assert_array_equal(h.applied_stress, [-3.4])

        # Loading function
        h.internal_torque = lambda element, t: 5.4
        h.calc_external_loading()
        assert_array_equal(h.applied_forces, 0)
        assert_array_equal(h.applied_stress, [-5.4])
