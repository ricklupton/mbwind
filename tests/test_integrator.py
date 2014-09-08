import unittest
import numpy as np
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal as assert_aae)
from mbwind import (System, Integrator, RigidConnection, Hinge,
                    RigidBody, StateOutput, LoadOutput, CustomOutput)


class TestStateOutput(unittest.TestCase):
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError, msg='deriv must be 0, 1, or 2'):
            StateOutput('state-name', 3)

    def test_label(self):
        self.assertEqual(StateOutput('state', label='label').label, 'label')
        self.assertEqual(StateOutput('state', deriv=0).label,
                         'state <state>')
        self.assertEqual(StateOutput('state', deriv=1).label,
                         'd/dt state <state>')
        self.assertEqual(StateOutput('state', deriv=2).label,
                         'd2/dt2 state <state>')
        self.assertEqual(StateOutput('state', deriv=2, local=True).label,
                         'd2/dt2 state <state> [local]')

    def test_call(self):
        s = System()
        c = RigidConnection('conn', [1, 0, 0])
        h = Hinge('hinge', [0, 1, 0])
        b = RigidBody('body', 1)
        s.add_leaf(h)
        h.add_leaf(c)
        c.add_leaf(b)
        s.setup()

        # Set hinge angle
        h.xstrain[0] = 0.82
        h.vstrain[0] = 1.2
        h.astrain[0] = -0.3
        s.update_kinematics()

        # Test node outputs
        out = StateOutput('node-1')
        assert_array_equal(out(s), np.r_[b.rp, b.Rp.flatten()])

        out = StateOutput('node-1', deriv=1)
        assert_array_equal(out(s), b.vp)

        out = StateOutput('node-1', deriv=2)
        assert_array_equal(out(s), b.ap)

        out = StateOutput('node-1', local=True)
        assert_array_equal(out(s), np.r_[np.dot(b.Rp.T, b.rp),
                                         np.eye(3).flatten()])

        out = StateOutput('node-1', deriv=1, local=True)
        assert_array_equal(out(s), np.r_[np.dot(b.Rp.T, b.vp[:3]),
                                         np.dot(b.Rp.T, b.vp[3:])])

        out = StateOutput('node-1', deriv=2, local=True)
        assert_array_equal(out(s), np.r_[np.dot(b.Rp.T, b.ap[:3]),
                                         np.dot(b.Rp.T, b.ap[3:])])

        # Test strain outputs
        out = StateOutput('hinge-strains')
        assert_array_equal(out(s), 0.82)

        out = StateOutput('hinge-strains', deriv=1)
        assert_array_equal(out(s), 1.2)

        out = StateOutput('hinge-strains', deriv=2)
        assert_array_equal(out(s), -0.3)

        # Strains cannot be transformed to local coordinates
        with self.assertRaises(RuntimeError):
            out = StateOutput('hinge-strains', local=True)
            out(s)


class TestLoadOutput(unittest.TestCase):
    def test_label(self):
        self.assertEqual(LoadOutput('state', label='label').label, 'label')
        self.assertEqual(LoadOutput('state').label, 'reaction load on <state>')
        self.assertEqual(LoadOutput('state', local=True).label,
                         'reaction load on <state> [local]')

    def test_call(self):
        s = System()
        c = RigidConnection('conn', [1, 0, 0])
        h = Hinge('hinge', [0, 1, 0])
        b = RigidBody('body', 1)
        s.add_leaf(h)
        h.add_leaf(c)
        c.add_leaf(b)
        s.setup()

        # Set hinge angle
        h.xstrain[0] = 0.82
        h.vstrain[0] = 1.2
        h.astrain[0] = -0.3
        s.update_kinematics()
        s.solve_reactions()

        # Test load outputs
        out = LoadOutput('node-1')
        assert_array_equal(out(s), s.joint_reactions['node-1'])

        out = LoadOutput('node-1', local=True)
        F = s.joint_reactions['node-1']
        assert_array_equal(out(s), np.r_[np.dot(b.Rp.T, F[:3]),
                                         np.dot(b.Rp.T, F[3:])])


class TestCustomOutput(unittest.TestCase):
    def test_label(self):
        def custom_func(system):
            pass
        self.assertEqual(CustomOutput(custom_func, label='label').label,
                         'label')
        self.assertRegex(CustomOutput(custom_func).label,
                         r'custom output <function TestCustomOutput.test_label'
                         r'.<locals>.custom_func at [0-9a-z]+>')

    def test_call(self):
        s = System()

        def custom_func(system):
            return "custom {}".format(system is s)
        out = CustomOutput(custom_func)
        self.assertEqual(out(s), 'custom True')
        self.assertTrue(isinstance(out(s), np.ndarray))


class TestIntegrator(unittest.TestCase):
    def test_simple_prescribed_integration(self):
        s = System()
        h = Hinge('hinge', [0, 1, 0])
        s.add_leaf(h)
        s.setup()

        s.prescribe(h)
        w = h.vstrain[0] = 0.97  # rad/s

        integ = Integrator(s)
        t, y = integ.integrate(9.0, 0.1)

        # Check time vector and shape of result
        assert_array_equal(t, np.arange(0, 9.0, 0.1))
        self.assertEqual(len(y), 1)
        self.assertEqual(y[0].shape, (len(t), 1))

        # Result should be y = wt, but wrapped to [0, 2pi)
        assert_aae(y[0][:, 0], (w * t) % (2 * np.pi))

        # Check asking for velocity and acceleration works
        h.xstrain[0] = s.time = 0.0  # reset
        integ = Integrator(s, ('pos', 'vel', 'acc'))
        t, y = integ.integrate(1.0, 0.1)
        assert_array_equal(t, np.arange(0, 1.0, 0.1))
        self.assertEqual(len(y), 3)
        for yy in y:
            self.assertEqual(yy.shape, (len(t), 1))
        assert_aae(y[0][:, 0], w * t)
        assert_aae(y[1][:, 0], w)
        assert_aae(y[2][:, 0], 0)

    def test_callback(self):
        s = System()
        s.setup()

        # Exponential decay: qd = -A q
        def callback(system, ti, q_struct, q_other):
            self.assertIs(system, s)
            self.assertEqual(len(q_other), 1)
            return -q_other

        integ = Integrator(s)
        t, y = integ.integrate(9.0, 0.1, extra_states=np.ones(1),
                               callback=callback)

        # Check time vector and shape of result
        assert_array_equal(t, np.arange(0, 9.0, 0.1))
        self.assertEqual(len(y), 1)
        self.assertEqual(y[0].shape, (len(t), 1))
        assert_aae(y[0][:, 0], np.exp(-t))


if __name__ == '__main__':
    unittest.main()
