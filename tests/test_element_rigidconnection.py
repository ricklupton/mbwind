import unittest
import numpy as np
from numpy import dot
from numpy.testing import assert_array_equal

from mbwind.elements import RigidConnection
from .common import (random_rotation_matrix, set_random_state,
                     constraint_consistency_check)


def _make_random_joint(rdm):
    # Make hinge with random axis and transform
    return RigidConnection('conn',
                           offset=random_rotation_matrix(rdm)[:, 0],
                           rotation=random_rotation_matrix(rdm))


class TestRigidConnection(unittest.TestCase):
    def setUp(self):
        self.rdm = np.random.RandomState(123456789)

    def test_defaults(self):
        conn = RigidConnection('conn')
        assert_array_equal(conn.offset, 0)
        assert_array_equal(conn.rotation, np.eye(3))

    def test_distal_node_position(self):
        NTESTS = 50
        for i in range(NTESTS):
            j = _make_random_joint(self.rdm)
            set_random_state(j, self.rdm, 1)

            # Test distal transform
            j.calc_distal_pos()
            assert_array_equal(j.rd, j.rp + dot(j.Rp, j.offset))
            assert_array_equal(j.Rd, dot(j.Rp, j.rotation))

    def test_velocity_and_acceleration_constraints_are_consistent(self):
        # Check for consistency with random conditions
        constraint_consistency_check(_make_random_joint, self.rdm)
