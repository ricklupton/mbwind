import unittest
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal as assert_aae)
from mbwind import System, LinearisedSystem, PrismaticJoint, RigidBody


class TestLinearisedSystem(unittest.TestCase):
    def test_1dof_linear_system(self):
        s = System()
        j = PrismaticJoint('joint', [0, 0, 1])
        j.stiffness = 5.0
        j.damping = 2.3
        b = RigidBody('body', 10)
        s.add_leaf(j)
        j.add_leaf(b)
        s.setup()

        linsys = LinearisedSystem.from_system(s)

        assert_array_equal(linsys.M, [[10.0]])
        assert_array_equal(linsys.C, [[2.3]])
        assert_array_equal(linsys.K, [[5.0]])

    def test_1dof_nonlinear_system(self):
        s = System()
        j = PrismaticJoint('joint', [0, 0, 1])
        k = 0.45  # quadratic stiffness coefficient
        j.internal_force = lambda el, t: -k * el.xstrain[0]**2
        b = RigidBody('body', 10)
        s.add_leaf(j)
        j.add_leaf(b)
        s.setup()

        # Linearise around z0 = 0: stiffness should be zero
        linsys = LinearisedSystem.from_system(s, z0=0)
        assert_aae(linsys.M, [[10.0]])
        assert_aae(linsys.C, [[0.0]])
        assert_aae(linsys.K, [[0.0]])

        # Linearise about z0 = 2: stiffness should be 2kx
        linsys = LinearisedSystem.from_system(s, z0=[2])
        assert_aae(linsys.M, [[10.0]])
        assert_aae(linsys.C, [[0.0]])
        assert_aae(linsys.K, [[2 * k * 2]])

        # Test setting z0 in another way
        linsys = LinearisedSystem.from_system(s, z0={'joint': [4.2]})
        assert_aae(linsys.M, [[10.0]])
        assert_aae(linsys.C, [[0.0]])
        assert_aae(linsys.K, [[2 * k * 4.2]])
