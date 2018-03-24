import unittest
import numpy as np
from numpy import linspace, sqrt, pi, dot
from numpy.testing import assert_array_almost_equal

from mbwind import (System, LoadOutput, CustomOutput, PrismaticJoint,
                    ModalElementFromFE, RigidBody, Integrator,
                    RigidConnection, Hinge, rotations)

from beamfe import BeamFE

assert_aae = assert_array_almost_equal


class TestReactionForcesOnModalElementFromFE(unittest.TestCase):
    """
    System
    ------
    A triangular rigid beam, offset by a rigid link from a hinge.

    Tests
    -----
    Set the angular acceleration of the hinge. Check the reaction
    forces at the centre and at the root of the beam.
    """
    mass = 5.0     # kg
    length = 20.0  # m
    offset = 3.2   # m
    force = -34.2  # N / m

    def setUp(self):
        # FE model for beam - no modes, i.e. rigid
        x = linspace(0, self.length, 20)
        density = (2 * self.mass / self.length) * (1 - x / self.length)
        fe = BeamFE(x, density=density, EA=0, EIy=1, EIz=0)
        fe.set_boundary_conditions('C', 'F')
        self.beam = ModalElementFromFE('beam', fe, 0)

        # Set loading - in Z direction
        load = np.zeros((len(x), 3))
        load[:, 2] = self.force
        self.beam.loading = load

        # Offset from hinge axis
        self.conn = RigidConnection('offset', [self.offset, 0, 0])

        # Hinge with axis along Y axis
        self.hinge = Hinge('hinge', [0, 1, 0])

        # Build system
        self.system = System()
        self.system.add_leaf(self.hinge)
        self.hinge.add_leaf(self.conn)
        self.conn.add_leaf(self.beam)
        self.system.setup()
        self.system.update_kinematics()    # Set up nodal values initially

    def test_reactions(self):
        # Set angular acceleration
        alpha = 1.235  # rad/s2
        self.hinge.astrain[0] = alpha
        self.system.update_kinematics()    # Update nodal values based on DOFs
        self.system.solve_reactions()      # Solve reactions incl d'Alembert

        # Some parameters
        L = self.length
        m = self.mass
        Ro = self.offset
        Rg = L / 3   # distance to CoM along beam
        IG = m * L ** 2 / 18
        assert_aae(m, self.beam.mass_vv[0, 0])

        # Check reactions at beam root
        P = self.system.joint_reactions['node-1']
        Fz_expected = (-m * (Ro + Rg) * alpha -
                       self.force * L)
        My_expected = ((IG + m * Rg * (Ro + Rg)) * alpha +
                       self.force * L ** 2 / 2)
        assert_aae(P, [0, 0, Fz_expected, 0, My_expected, 0])


class TestPendulumReactionLoadsTimeseries(unittest.TestCase):
    def setUp(self):
        # Parameters
        mass = 11.234
        length = 2.54
        gravity = 9.81

        # Build model
        hinge = Hinge('hinge', [0, 1, 0])
        link = RigidConnection('link', [length, 0, 0])
        body = RigidBody('body', mass)

        system = System(gravity=gravity)
        system.add_leaf(hinge)
        hinge.add_leaf(link)
        link.add_leaf(body)
        system.setup()

        # Custom outputs to calculate correct answer
        def force_body_prox_local(s):
            theta = s.q[hinge.istrain][0]
            thetadot = s.qd[hinge.istrain][0]
            thetadotdot = s.qdd[hinge.istrain][0]
            Fx = mass * (-gravity*np.sin(theta) - length*thetadot**2)
            Fz = mass * (+gravity*np.cos(theta) - length*thetadotdot)
            return [Fx, 0, Fz, 0, 0, 0]

        def force_hinge_prox(s):
            theta = s.q[hinge.istrain][0]
            thetadot = s.qd[hinge.istrain][0]
            thetadotdot = s.qdd[hinge.istrain][0]
            A = np.array([[+np.cos(theta), np.sin(theta)],
                          [-np.sin(theta), np.cos(theta)]])
            Fxz = -mass * length * np.dot(A, [thetadot**2, thetadotdot])
            return [Fxz[0], 0, Fxz[1] + gravity*mass, 0, 0, 0]

        # Solver
        integ = Integrator(system, ('pos', 'vel', 'acc'))
        integ.add_output(LoadOutput(hinge.iprox))
        integ.add_output(LoadOutput(link.iprox))
        integ.add_output(LoadOutput(body.iprox))
        integ.add_output(LoadOutput(body.iprox, local=True))
        integ.add_output(CustomOutput(force_hinge_prox, "correct ground"))
        integ.add_output(CustomOutput(force_body_prox_local,
                                      "correct link distal local"))

        self.system = system
        self.integ = integ

    def test_results(self):
        t, y = self.integ.integrate(3.8, 0.03)

        # reaction forces vs predictions - base
        assert_aae(y[3], y[7])

        # reaction forces vs predictions - local body prox
        assert_aae(y[6], y[8])


class TestSliderReactionLoadsTimeseries(unittest.TestCase):
    def setUp(self):
        # Parameters
        mass = 11.234
        length = 2.54
        gravity = 9.81

        # Build model
        slider = PrismaticJoint('slider', [1, 0, 0])
        link = RigidConnection('link', [0, 0, length])
        body = RigidBody('body', mass)

        system = System(gravity=gravity)
        system.add_leaf(slider)
        slider.add_leaf(link)
        link.add_leaf(body)
        system.setup()

        # Prescribe motion -- sinusoidal acceleration
        motion_frequency = 1    # Hz
        motion_amplitude = 2.3  # m

        # x =  motion_amplitude * np.cos(w*t)
        # v = -motion_amplitude * np.sin(w*t) * w
        # a = -motion_amplitude * np.cos(w*t) * w**2
        def prescribed_acceleration(t):
            w = 2*np.pi*motion_frequency
            return -w**2 * motion_amplitude * np.cos(w*t)

        system.prescribe(slider, prescribed_acceleration)

        # Set the correct initial condition
        system.q[slider.istrain][0] = motion_amplitude
        system.qd[slider.istrain][0] = 0.0

        # Custom outputs to calculate correct answer
        def force_body_prox(s):
            a = prescribed_acceleration(s.time)
            Fx = mass * a
            Fz = mass * gravity
            return [Fx, 0, Fz, 0, 0, 0]

        def force_link_prox(s):
            a = prescribed_acceleration(s.time)
            Fx = mass * a
            Fz = mass * gravity
            My = length * Fx
            return [Fx, 0, Fz, 0, My, 0]

        def force_slider_prox(s):
            a = prescribed_acceleration(s.time)
            x = -a / (2*np.pi*motion_frequency)**2
            Fx = mass * a
            Fz = mass * gravity
            My = length*Fx - x*Fz
            return [Fx, 0, Fz, 0, My, 0]

        # Solver
        integ = Integrator(system, ('pos', 'vel', 'acc'))
        integ.add_output(LoadOutput(slider.iprox))
        integ.add_output(LoadOutput(link.iprox))
        integ.add_output(LoadOutput(body.iprox))
        integ.add_output(CustomOutput(force_slider_prox, "correct ground"))
        integ.add_output(CustomOutput(force_link_prox, "correct slider dist"))
        integ.add_output(CustomOutput(force_body_prox, "correct link dist"))

        self.system = system
        self.integ = integ

    def test_results(self):
        t, y = self.integ.integrate(3.8, 0.02)

        # reaction forces vs predictions - base
        results = y[3:6]
        predictions = y[6:9]
        for i in range(3):
            assert_aae(results[i], predictions[i])


class TestReactionForcesForCentrifugalForce(unittest.TestCase):
    """
    System
    ------
    A rigid body with offset mass, attached to a spinning hinge.

    Tests
    -----
    Check centrifugal force reaction on hinge is in correct direction.
    """
    mass = 5.0     # kg
    offset = 3.2   # m

    def setUp(self):
        # Rigid body with offset centre of mass
        self.body = RigidBody('body', self.mass, Xc=[self.offset, 0, 0])

        # Hinge with axis along Z axis
        self.hinge = Hinge('hinge', [0, 0, 1])

        # Build system
        self.system = System()
        self.system.add_leaf(self.hinge)
        self.hinge.add_leaf(self.body)
        self.system.setup()
        self.system.update_kinematics()    # Set up nodal values initially
        self.system.update_matrices()

    def test_reactions(self):
        # Set angular acceleration
        w = 5.21  # rad/s
        self.hinge.vstrain[0] = w
        self.system.update_kinematics()    # Update nodal values based on DOFs
        self.system.update_matrices()
        self.system.solve_reactions()      # Solve reactions incl d'Alembert

        # Some parameters
        L = self.offset
        m = self.mass

        # Check reactions at beam root
        Pg = self.system.joint_reactions['ground']
        P0 = self.system.joint_reactions['node-0']
        Fx_expected = -m * L * w**2
        assert_aae(P0, [Fx_expected, 0, 0, 0, 0, 0])
        assert_aae(Pg, P0)


class TestReactionForcesWithRotatedBeam(unittest.TestCase):
    """Intended to check the transformation from blade loading to rotor
    loading in a wind turbine rotor: the loads are applied to the beam
    in the local rotated coordinate system, check they work through to
    the ground reactions correctly.
    """
    force = 24.1
    length = 4.3
    root_length = 0.0

    def setUp(self):
        # FE model for beam - no modes, i.e. rigid
        self.x = x = linspace(0, self.length, 20)
        fe = BeamFE(x, density=2, EA=0, EIy=0, EIz=0)

        # Build the elements
        self.shaft = Hinge('shaft', [1, 0, 0])

        self.roots = []
        self.blades = []
        self.pitch_bearings = []
        for ib in range(1):
            R = rotations(('x', ib*2*pi/3), ('y', -pi/2))
            root_offset = dot(R, [self.root_length, 0, 0])
            root = RigidConnection('root%d' % (ib+1), root_offset, R)
            bearing = Hinge('pitch%d' % (ib+1), [1, 0, 0])
            blade = ModalElementFromFE('blade%d' % (ib+1), fe, 0)

            self.shaft.add_leaf(root)
            root.add_leaf(bearing)
            bearing.add_leaf(blade)

            self.roots.append(root)
            self.blades.append(blade)
            self.pitch_bearings.append(bearing)

        # Build system
        self.system = System()
        self.system.add_leaf(self.shaft)
        self.system.setup()
        self.system.update_kinematics()    # Set up nodal values initially
        self.system.update_matrices()

    def test_reactions(self):
        # Some parameters
        L = self.length
        F = self.length * self.force

        # Set loading - in local z direction
        load = np.zeros((len(self.x), 3))
        load[:, 2] = self.force
        self.blades[0].loading = load
        self.system.update_kinematics()
        self.system.update_matrices()
        self.system.solve_reactions()

        # Check reactions at ground (0, 0, 0)
        P = -self.system.joint_reactions['ground']
        F_expected = [-F, 0, 0]
        M_expected = [0, -F*(L+self.root_length)/2, 0]
        assert_aae(P, np.r_[F_expected, M_expected])

        # Reactions on other side of hinge
        P2 = -self.system.joint_reactions['node-0']
        assert_aae(P, P2)

        # Now set pitch angle to 45deg
        # NB: hinge rotation is opposite to wind turbine pitch convention
        self.pitch_bearings[0].xstrain[0] = -pi / 4
        self.system.update_kinematics()
        self.system.update_matrices()
        self.system.solve_reactions()

        # Check reactions at ground (0, 0, 0)
        P = -self.system.joint_reactions['ground']
        F_expected = [-F/sqrt(2), F/sqrt(2), 0]
        M_expected = [-F/sqrt(2)*L/2, -F/sqrt(2)*L/2, 0]
        assert_aae(P, np.r_[F_expected, M_expected])

        # Reactions on other side of hinge
        P2 = -self.system.joint_reactions['node-0']
        assert_aae(P, P2)

