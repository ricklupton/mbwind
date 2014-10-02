import unittest
import numpy as np
from numpy import dot
from numpy.testing import assert_array_equal

from mbwind.elements import PrismaticJoint
from .common import (random_rotation_matrix, set_random_state,
                     constraint_consistency_check)


def _make_random_joint(rdm):
    # Make hinge with random axis and transform
    return PrismaticJoint('joint',
                          axis=random_rotation_matrix(rdm)[:, 0],
                          post_transform=random_rotation_matrix(rdm))


class TestPrismaticJoint(unittest.TestCase):
    def setUp(self):
        self.rdm = np.random.RandomState(123456789)

    def test_distal_node_position(self):
        NTESTS = 50
        for i in range(NTESTS):
            j = _make_random_joint(self.rdm)
            set_random_state(j, self.rdm, 1)

            # Test distal transform
            j.calc_distal_pos()
            assert_array_equal(j.rd, j.rp + j.xstrain[0] * dot(j.Rp, j.axis))
            assert_array_equal(j.Rd, dot(j.Rp, j.post_transform))

    def test_velocity_and_acceleration_constraints_are_consistent(self):
        # Check for consistency with random conditions
        constraint_consistency_check(_make_random_joint, self.rdm)

    def test_interal_force(self):
        # NB minus sign because convention for applied_stress is that
        # stiffness loads are positive.
        j = PrismaticJoint('joint', [0, 0, 1])

        # Constant loading
        j.internal_force = 3.4
        j.calc_external_loading()
        assert_array_equal(j.applied_forces, 0)
        assert_array_equal(j.applied_stress, [-3.4])

        # Loading function
        j.internal_force = lambda element, t: 5.4
        j.calc_external_loading()
        assert_array_equal(j.applied_forces, 0)
        assert_array_equal(j.applied_stress, [-5.4])
