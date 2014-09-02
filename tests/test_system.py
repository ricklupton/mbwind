import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal as assert_aae
from mbwind import (System, RigidConnection, RigidBody,
                    PrismaticJoint, FreeJoint)


class TestSystem(unittest.TestCase):
    def test_adding_elements(self):
        conn = RigidConnection('conn')
        body = RigidBody('body', mass=1.235)
        s = System()
        s.add_leaf(conn)
        conn.add_leaf(body)
        s.setup()

        # Should have dict of elements
        self.assertEqual(s.elements, {'conn': conn, 'body': body})

        # Number of states:
        #   6 ground
        # + 6 constraints on conn
        # + 6 <node-0>   between conn and body
        # ---
        #  18
        self.assertEqual(s.lhs.shape, (18, 18))
        for vec in (s.rhs, s.qd, s.qdd):
            self.assertEqual(len(vec), 18)

        # Check there are no dofs
        self.assertEqual(len(s.q.dofs), 0)
        self.assertEqual(len(s.qd.dofs), 0)
        self.assertEqual(len(s.qdd.dofs), 0)

    def test_adding_elements_with_strains(self):
        slider = PrismaticJoint('slider', [1, 0, 0])
        conn = RigidConnection('conn')
        body = RigidBody('body', mass=1.235)
        s = System()
        s.add_leaf(slider)
        slider.add_leaf(conn)
        conn.add_leaf(body)
        s.setup()

        # Should have dict of elements
        self.assertEqual(s.elements,
                         {'slider': slider, 'conn': conn, 'body': body})

        # Number of states:
        #   6 ground
        # + 6 constraints on slider
        # + 1 strain in slider
        # + 6 <node-0>   between slider and conn
        # + 6 constraints on conn
        # + 6 <node-1>   between conn and body
        # ---
        #  31
        self.assertEqual(s.lhs.shape, (31, 31))
        for vec in (s.rhs, s.qd, s.qdd):
            self.assertEqual(len(vec), 31)

        # Check there is the one slider dof
        self.assertEqual(len(s.q.dofs), 1)
        self.assertEqual(len(s.qd.dofs), 1)
        self.assertEqual(len(s.qdd.dofs), 1)

        # After prescribing the slider, there should be no dofs
        s.prescribe(slider)
        self.assertEqual(len(s.q.dofs), 0)
        self.assertEqual(len(s.qd.dofs), 0)
        self.assertEqual(len(s.qdd.dofs), 0)

    def test_iter_elements(self):
        #     /-- c2
        #  c1-|
        #     \-- c3 --- c4
        s = System()
        c1 = RigidConnection('c1')
        c2 = RigidConnection('c2')
        c3 = RigidConnection('c3')
        c4 = RigidConnection('c4')
        s.add_leaf(c1)
        c1.add_leaf(c2)
        c1.add_leaf(c3)
        c3.add_leaf(c4)
        s.setup()

        # Should iter elements depth-first
        self.assertEqual(list(s.iter_elements()),
                         [c1, c2, c3, c4])

    def test_get_set_state(self):
        s = System()
        j = FreeJoint('joint')
        s.add_leaf(j)
        s.setup()

        # State is [q_dofs, qd_dofs].
        # Here we have 6 dofs:
        s.set_state([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        self.assertEqual(list(s.q.dofs), [1, 2, 3, 4, 5, 6])
        self.assertEqual(list(s.qd.dofs), [7, 8, 9, 10, 11, 12])

        s.q.dofs[2] = 100
        s.qd.dofs[0] = -1
        self.assertEqual(list(s.get_state()),
                         [1, 2, 100, 4, 5, 6, -1, 8, 9, 10, 11, 12])

    def test_prescribe_free(self):
        s = System()
        j = FreeJoint('joint')
        s.add_leaf(j)
        s.setup()

        # Initially all 6 joint motions are free
        self.assertEqual(len(s.q.dofs), 6)

        # Prescribing joint to be fixed results in no dofs
        s.prescribe(j)
        self.assertEqual(len(s.q.dofs), 0)

        # Freeing joint results in 6 dofs again
        s.free(j)
        self.assertEqual(len(s.q.dofs), 6)

        # Prescribing 2 of joint motions leaves 4 dofs
        s.prescribe(j, lambda t: t, [0, 2])
        self.assertEqual(len(s.q.dofs), 4)

        # Prescribing another joint motions leaves 3 dofs
        s.prescribe(j, 2.0, part=3)
        self.assertEqual(len(s.q.dofs), 3)

        # Check accelerations are applied to qdd
        assert_aae(s.qdd[j.istrain], 0)
        s.apply_prescribed_accelerations(time=3.54)
        assert_aae(s.qdd[j.istrain], [3.54, 0, 3.54, 2.0, 0, 0])

        # Freeing joint results in 6 dofs again
        s.free(j)
        self.assertEqual(len(s.q.dofs), 6)

    def test_assemble(self):
        # Test system:
        #
        #                   /-- m2
        #  m1 -----conn----|
        #                   \-- m3
        s = System()
        m1 = RigidBody('m1', 1.3)
        m2 = RigidBody('m2', 3.4)
        m3 = RigidBody('m3', 7.5)
        conn = RigidConnection('conn')
        s.add_leaf(conn)
        s.add_leaf(m1)
        conn.add_leaf(m2)
        conn.add_leaf(m3)
        s.setup()

        # Check starting mass matrices of elements are as expected
        assert_aae(np.diag(m1.mass_vv[:3, :3]), 1.3)
        assert_aae(np.diag(m2.mass_vv[:3, :3]), 3.4)
        assert_aae(np.diag(m3.mass_vv[:3, :3]), 7.5)

        # Initially system matrix is empty
        assert_aae(s.lhs, 0)

        # After assembly, the mass matrices are put in the correct places:
        #   0:6  -> m1 node
        #   6:12 -> conn constraints
        #  12:18 -> m2 node
        #  12:18 -> m3 node
        s.assemble()
        M = s.lhs.copy()
        # Subtract expected mass
        M[0:6, 0:6] -= m1.mass_vv
        M[12:18, 12:18] -= m2.mass_vv + m3.mass_vv
        # Subtract expected constraints
        M[0:6, 6:12] -= conn.F_vp
        M[6:12, 0:6] -= conn.F_vp
        M[12:18, 6:12] -= conn.F_vd
        M[6:12, 12:18] -= conn.F_vd
        assert_aae(M, 0)

    def test_update_kinematics(self):
        assert False


if __name__ == '__main__':
    unittest.main()
