import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal as assert_aae
from mbwind import (System, RigidConnection, RigidBody, Hinge,
                    PrismaticJoint, FreeJoint)


class TestSystem(unittest.TestCase):
    def test_print_functions(self):
        # Not very good tests, but at least check they run without errors
        joint = FreeJoint('joint')
        body = RigidBody('body', mass=1.235)
        s = System()
        s.add_leaf(joint)
        joint.add_leaf(body)
        s.setup()
        s.print_states()
        s.print_info()

    def test_dofs_subset(self):
        s = System()
        j = FreeJoint('joint')
        s.add_leaf(j)
        s.setup()

        # 2 nodes, 6 constraints, 6 dofs
        self.assertEqual(len(s.q), 2 * 12 + 6 + 6)
        self.assertEqual(len(s.qd), 2 * 6 + 6 + 6)
        self.assertEqual(len(s.qdd), 2 * 6 + 6 + 6)
        self.assertEqual(len(s.q.dofs), 6)
        self.assertEqual(len(s.qd.dofs), 6)
        self.assertEqual(len(s.qdd.dofs), 6)

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
        s.time = 3.54

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
        s.apply_prescribed_accelerations()
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

        # Initially make system matrix empty for testing
        s.lhs[:, :] = 0
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

    def test_update_kinematics_results(self):
        # Test system: (all rigid connections of length 1)
        #
        #                  [hinge]
        #  (gnd)---c1---(0)
        #               (1)---c2---(2)
        #                           |
        #  y                        c3
        #  |                        |
        #  |--> x                  (3)
        #
        s = System()
        c1 = RigidConnection('c1', [1, 0, 0])
        c2 = RigidConnection('c2', [1, 0, 0])
        c3 = RigidConnection('c3', [0, -1, 0])
        hinge = Hinge('hinge', [0, 0, 1])
        s.add_leaf(c1)
        c1.add_leaf(hinge)
        hinge.add_leaf(c2)
        c2.add_leaf(c3)
        s.setup()

        # All velocities and accelerations should be zero
        for el in [c1, c2, c3, hinge]:
            assert_aae(el.vp, 0)
            assert_aae(el.vd, 0)
            assert_aae(el.ap, 0)
            assert_aae(el.ad, 0)

        # (gnd)
        assert_aae(c1.rp, 0)
        assert_aae(c1.Rp, np.eye(3))

        # (0)
        assert_aae(c1.rd, [1, 0, 0])
        assert_aae(c1.Rd, np.eye(3))

        # (1)
        assert_aae(c2.rp, [1, 0, 0])
        assert_aae(c2.Rp, np.eye(3))

        # (2)
        assert_aae(c3.rp, [2, 0, 0])
        assert_aae(c3.Rp, np.eye(3))

        # (3)
        assert_aae(c3.rd, [2, -1, 0])
        assert_aae(c3.Rd, np.eye(3))

        ##### now set angular velocity of hinge #####
        hinge.vstrain[0] = 1.0
        s.update_kinematics()

        # (gnd)
        assert_aae(c1.vp, 0)
        assert_aae(c1.ap, 0)

        # (0)
        assert_aae(c1.vd, 0)
        assert_aae(c1.ad, 0)

        # (1)
        assert_aae(c2.vp, [0, 0, 0, 0, 0, 1.0])
        assert_aae(c2.ap, 0)

        # (2)
        assert_aae(c3.vp, [0, 1, 0, 0, 0, 1.0])
        assert_aae(c3.ap, [-1, 0, 0, 0, 0, 0])   # centripetal acceleration

        # (3)
        assert_aae(c3.vd, [1, 1, 0, 0, 0, 1.0])
        assert_aae(c3.ad, [-1, 1, 0, 0, 0, 0])  # centripetal acceleration

    def test_solve_accelerations(self):
        # solve_accelerations() should find:
        #  (a) response of system to forces (here, gravity)
        #  (b) include any prescribed accelerations in qdd vector
        g = 9.81
        s = System(gravity=g)
        j = FreeJoint('joint')
        b = RigidBody('body', mass=23.54, inertia=52.1 * np.eye(3))
        s.add_leaf(j)
        j.add_leaf(b)
        s.setup()

        # Prescribe horizontal acceleration. Vertical acceleration
        # should result from gravity.
        s.prescribe(j, 2.3, part=[0])  # x acceleration

        # Initially accelerations are zero
        assert_aae(j.ap, 0)
        assert_aae(j.ad, 0)
        assert_aae(j.astrain, 0)

        # Solve accelerations & check
        s.solve_accelerations()
        s.update_kinematics()
        assert_aae(j.ap, 0)  # ground
        assert_aae(j.ad, [2.3, 0, -g, 0, 0, 0])
        assert_aae(j.astrain, j.ad)  # not always true, but works for FreeJoint

    def test_solve_accelerations_coupling(self):
        # Further to test above, check that coupling between prescribed
        # accelerations and other dofs is correct. For example, if there
        # is a rigid body vertically offset from the joint, then a
        # prescribed horizontal acceleration should cause an angular
        # acceleration as well as the translational acceleration.
        s = System()
        j = FreeJoint('joint')
        c = RigidConnection('conn', [0, 0, 1.7])
        b = RigidBody('body', mass=23.54, inertia=74.1 * np.eye(3))
        s.add_leaf(j)
        j.add_leaf(c)
        c.add_leaf(b)
        s.setup()

        # Prescribe horizontal acceleration, solve other accelerations
        s.prescribe(j, 2.3, part=[0])  # x acceleration
        s.update_kinematics()          # update system to show prescribed acc
        s.solve_accelerations()        # solve free accelerations
        s.update_kinematics()          # update system to show solution

        # Ground shouldn't move
        assert_aae(j.ap, 0)

        # Need angular acceleration = (m a_x L) / I0
        I0 = 74.1 + (23.54 * 1.7**2)
        expected_angular_acc = -(23.54 * 2.3 * 1.7) / I0
        assert_aae(j.ad, [2.3, 0, 0, 0, expected_angular_acc, 0])
        assert_aae(j.astrain, j.ad)  # not always true, but works for FreeJoint

    def test_solve_reactions(self):
        # Check it calls the Element method in the right order: down
        # the tree from leaves to base. It must also reset reactions.
        s = System()
        c0 = RigidConnection('c0')
        c1 = RigidConnection('c1')
        c2 = RigidConnection('c2')
        b1 = RigidBody('b1', 1)
        b2 = RigidBody('b2', 1)
        s.add_leaf(c0)
        c0.add_leaf(c1)
        c0.add_leaf(c2)
        c1.add_leaf(b1)
        c2.add_leaf(b2)
        s.setup()

        # Check elements' iter_reactions() are called
        def mock_iter_reactions(element):
            calls.append(element)
        calls = []
        import types
        for el in s.elements.values():
            el.iter_reactions = types.MethodType(mock_iter_reactions, el)

        # Test
        s.joint_reactions[:] = 3
        s.solve_reactions()
        self.assertEqual(calls, [b2, c2, b1, c1, c0])
        assert_aae(s.joint_reactions, 0)

    def test_find_equilibrium(self):
        g = 9.81
        m = 23.1
        k = 45.2
        s = System(gravity=g)
        slider = PrismaticJoint('slider', [0, 0, 1])
        slider.stiffness = k
        body = RigidBody('body', mass=m)
        s.add_leaf(slider)
        slider.add_leaf(body)
        s.setup()

        # Initially position should be zero and acceleration nonzero
        s.solve_accelerations()
        assert_aae(slider.xstrain, 0)
        assert_aae(slider.astrain, -g)

        # At equilibrium, position should be nozero and force on body zero
        s.find_equilibrium()
        s.update_matrices()      # recalculate stiffness force
        s.solve_accelerations()
        assert_aae(slider.xstrain, -m * g / k)
        assert_aae(slider.astrain, 0)

    def test_dof_index(self):
        s = System()
        j1 = FreeJoint('j1')
        j2 = FreeJoint('j2')
        s.add_leaf(j1)
        j1.add_leaf(j2)
        s.setup()

        # Prescribe some of the strains:
        #          _____j1____   _____j2____
        # Strain:  0 1 2 3 4 5   0 1 2 3 4 5
        # Prescr:    *     *             *
        # Dofs:    0   1 2   3   4 5 6   7 8
        s.prescribe(j1, 0, part=[1, 4])
        s.prescribe(j2, 0, part=[3])

        self.assertEqual(s.dof_index('j1', 0), 0)
        self.assertEqual(s.dof_index('j1', 2), 1)
        self.assertEqual(s.dof_index('j1', 5), 3)
        self.assertEqual(s.dof_index('j2', 5), 8)

        # Strain index out of range should give IndexError
        with self.assertRaises(IndexError):
            s.dof_index('j1', 6)

        # Asking for index of prescribed strain should give ValueError
        with self.assertRaises(ValueError):
            s.dof_index('j1', 1)


if __name__ == '__main__':
    unittest.main()
