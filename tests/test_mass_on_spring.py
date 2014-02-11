from nose.tools import *
import numpy as np
from numpy import zeros, array, eye, pi, dot, sqrt, c_, diag, cos, sin
from numpy import linalg
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_allclose)

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

    def test_step_response(self):
        # Calculate system parameters
        zeta = self.damping_coeff
        wn = np.sqrt(self.K / self.M)
        wd = wn * np.sqrt(1 - zeta**2)
        psi = np.arcsin(zeta)

        # Forcing - step function
        force_amp = 3.4
        t0 = 2.3
        force_func = lambda t: (force_amp if t >= t0 else 0)

        def callback(system, time, q_other):
            self.body.nodal_load = [0, 0, force_func(time)]
            return []
        t, solution = integrate(self.system, callback)

        decay = np.exp(-zeta * wn * (t - t0))
        expected_defl = 1 - decay * cos(wd * (t - t0) - psi) / cos(psi)
        X = force_amp / self.K
        expected_defl[t < t0] = 0
        assert_allclose(solution[:, 0], X * expected_defl, atol=1e-8)


def integrate(system, callback=None):
    # dopri5 gives exact result for step response
    integ = Integrator(system, method='dopri5')
    t, y = integ.integrate(10, 0.5, nprint=None, callback=callback)
    return t, y[0]
