import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal as assert_aae
from mbwind import (System, ReducedSystem, FreeJoint, RigidBody, Hinge,
                    PrismaticJoint, RigidConnection)


class TestReducedSystem(unittest.TestCase):
    def test_rigid_body_with_no_dofs(self):
        s = System()
        b = RigidBody('body', 23.7)
        s.add_leaf(b)
        s.setup()

        # Calculate reduced system to get rigid body matrices
        rsys = ReducedSystem(s)
        self.assertEqual(rsys.M.shape, (0, 0))
        self.assertEqual(rsys.Q.shape, (0,))

    def test_single_rigid_body(self):
        mass = 36.2
        inertia = np.diag([75.4, 653, 234])
        s = System()
        j = FreeJoint('joint')
        b = RigidBody('body', mass, inertia)
        s.add_leaf(j)
        j.add_leaf(b)
        s.setup()

        # Calculate reduced system to get rigid body matrices
        rsys = ReducedSystem(s)
        self.assertEqual(rsys.M.shape, (6, 6))
        self.assertEqual(rsys.Q.shape, (6,))
        assert_aae(rsys.M[:3, :3], mass * np.eye(3))
        assert_aae(rsys.M[3:, 3:], inertia)
        assert_aae(rsys.M[3:, :3], 0)
        assert_aae(rsys.M[:3, 3:], 0)
        assert_aae(rsys.Q, 0)

        # Now if some freedoms are prescribed, don't appear in reduced system
        s.prescribe(j, part=[1, 2, 3, 4, 5])  # only x-translation is free now
        rsys = ReducedSystem(s)
        self.assertEqual(rsys.M.shape, (1, 1))
        self.assertEqual(rsys.Q.shape, (1,))
        assert_aae(rsys.M[0, 0], mass)
        assert_aae(rsys.Q, 0)

    def test_applied_force(self):
        # Set up a hinge with a mass offset on a rigid body. The
        # reduced system should have 1 DOF -- the hinge rotation --
        # with the associated mass being the inertia of the mass about
        # the hinge, and the associated generalised force being the
        # applied moment.
        mass = 36.2
        zforce = -30
        L = 3.2
        s = System()
        h = Hinge('hinge', [0, 1, 0])
        c = RigidConnection('conn', [L, 0, 0])
        b = RigidBody('body', mass, nodal_load=[0, 0, zforce])
        s.add_leaf(h)
        h.add_leaf(c)
        c.add_leaf(b)
        s.setup()

        rsys = ReducedSystem(s)
        self.assertEqual(rsys.M.shape, (1, 1))
        self.assertEqual(rsys.Q.shape, (1,))
        self.assertEqual(rsys.M[0, 0], mass * L**2)  # inertial about hinge
        self.assertEqual(rsys.Q[0], -zforce * L)  # moment about hinge

    def test_nonzero_prescribed_acceleration(self):
        # Test reduction where a prescribed acceleration is non-zero:
        # two sliders in series, with a mass on the end. If the second
        # slider's acceleration is prescribed, the first slider's DOF
        # sees an inertial force corresponding to the acceleration of
        # the mass.
        mass = 36.2
        s = System()
        s1 = PrismaticJoint('slider1', [1, 0, 0])
        s2 = PrismaticJoint('slider2', [1, 0, 0])
        b = RigidBody('body', mass)
        s.add_leaf(s1)
        s1.add_leaf(s2)
        s2.add_leaf(b)
        s.setup()

        s.prescribe(s2, acc=0)

        # With hinge angle = 0, no generalised inertial force
        rsys = ReducedSystem(s)
        assert_aae(rsys.M, mass)
        assert_aae(rsys.Q, 0)

        # With hinge angle = 90deg, do see generalised inertial force
        s.prescribe(s2, acc=2.3)
        rsys = ReducedSystem(s)
        assert_aae(rsys.M, mass)
        assert_aae(rsys.Q, -mass * 2.3)


if __name__ == '__main__':
    unittest.main()
