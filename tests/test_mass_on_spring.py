from nose.tools import *
import numpy as np
from numpy import zeros, array, eye, pi, dot, sqrt, c_, diag, cos, sin
from numpy import linalg
from numpy.testing import assert_array_equal, assert_array_almost_equal

from mbwind.utils import rotmat_x, rotmat_y, rotmat_z, update_skewmat
from mbwind import RigidBody, PrismaticJoint, System, Integrator


class MassOnSpring_Tests:
    K = 53.2
    M = 23.1
    damping_coeff = 0.2

    def setup(self):
        joint = PrismaticJoint('joint', [0, 0, 1])
        joint.stiffness = self.K
        joint.damping = 2 * self.damping_coeff * (self.K * self.M) ** 0.5

        body = RigidBody('body', self.M)

        system = System()
        system.add_leaf(joint)
        joint.add_leaf(body)
        system.setup()
        system.update_kinematics()

        self.joint, self.body, self.system = joint, body, system

    def test_number_of_states(self):
        # 2 nodes       -> 12 states
        # 6 constraints -> 6 states
        # 1 dof         -> 1 state
        eq_(len(self.system.qd), 19)
        eq_(len(self.system.qd.dofs), 1)

    def test_solution_without_force(self):
        # Static equilibrium should not move with no forces
        self.system.find_equilibrium()
        assert_array_equal(self.system.q.dofs[:], 0)
        assert_array_equal(self.body.rp, 0)

        t, solution = integrate(self.system)
        assert_array_equal(solution, 0)

    def test_solution_with_force(self):
        Fz = -2.3
        self.body.nodal_load = [0, 0, Fz]

        self.system.find_equilibrium()
        assert_array_equal(self.system.q.dofs[:], Fz / self.K)
        assert_array_equal(self.body.rp, [0, 0, Fz / self.K])

        # Now integrating in time should not change solution
        t, solution = integrate(self.system)
        assert_array_equal(solution[0], solution[-1])

    def test_solution_with_time_varying_force(self):
        # Calculate starting position
        Fz = 3.4
        self.system.q.dofs[:] = -Fz / self.K
        force_func = lambda t: 3.4 * sin(2*pi*t/20)

        def callback(system, time, q_other):
            self.body.nodal_load = [0, 0, force_func(time)]
            print(time, self.body.nodal_load)
            return []
        t, solution = integrate(self.system, callback)

        expected_defl = -cos(2*pi*t/20) * Fz / self.K
        assert_array_equal(solution[:, 0], expected_defl)


def integrate(system, callback=None):
    integ = Integrator(system)
    t, y = integ.integrate(10, 0.5, nprint=None, callback=callback)
    return t, y[0]
