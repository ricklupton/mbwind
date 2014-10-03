import unittest
import numpy as np
from numpy import zeros, eye, pi, dot, sqrt, diag, ones_like, arange, linspace
from numpy.testing import assert_array_almost_equal as assert_aae
from mbwind import System, ReducedSystem
from mbwind.elements import (ModalElementFromFE, RigidConnection,
                             FreeJoint, Hinge)
from mbwind.utils import rotations
from beamfe import BeamFE


def _mock_rigid_uniform_beam(density, length, name='beam'):
    """Return empty modal representation of 20m long rigid beam"""
    x = linspace(0, length, 20)
    fe = BeamFE(x, density=density, EA=0, EIy=1, EIz=0)
    fe.set_boundary_conditions('C', 'F')
    modal = fe.modal_matrices(0)
    beam = ModalElementFromFE(name, modal)
    return beam


class TestModalElement(unittest.TestCase):
    def test_inertia_when_offset_axially(self):
        density = 230.4
        length = 20.0
        offset = 5.0
        element = _mock_rigid_uniform_beam(density, length)
        conn = RigidConnection('offset', offset=[offset, 0, 0])
        joint = FreeJoint('joint')

        system = System()
        system.add_leaf(joint)
        joint.add_leaf(conn)
        conn.add_leaf(element)
        system.setup()

        # Calculate reduced system to get rigid body matrices
        rsys = ReducedSystem(system)

        # Expected values: rod along x axis
        m = density * length
        Iy = m * (length**2 / 12 + (length/2 + offset)**2)
        expected_mass = m * eye(3)
        expected_inertia = diag([0, Iy, Iy])
        expected_offdiag = zeros((3, 3))

        # Y accel -> positive moment about Z
        # Z accel -> negative moment about Y
        expected_offdiag[2, 1] = +m * (length/2 + offset)
        expected_offdiag[1, 2] = -m * (length/2 + offset)

        assert_aae(rsys.M[:3, :3], expected_mass)
        assert_aae(rsys.M[3:, 3:], expected_inertia)
        assert_aae(rsys.M[3:, :3], expected_offdiag)
        assert_aae(rsys.M[:3, 3:], expected_offdiag.T)

    def test_three_rigid_elements_as_disc_have_ends_in_right_place(self):
        length = 20.0
        offset = 5.0

        # Make 3 elements spaced by 120 deg about z axis
        system = System()
        elements = []
        for i in range(3):
            rotmat = rotations(('z', i * 2*pi/3))
            offset_vector = dot(rotmat, [offset, 0, 0])
            conn = RigidConnection('offset%d' % i, offset_vector, rotmat)
            element = RigidConnection('element%d' % i, [length, 0, 0])
            elements.append(element)
            system.add_leaf(conn)
            conn.add_leaf(element)
        system.setup()

        r = offset
        R = offset + length
        assert_aae(elements[0].rp, [r,     0,           0])
        assert_aae(elements[1].rp, [-r/2,  r*sqrt(3)/2, 0])
        assert_aae(elements[2].rp, [-r/2, -r*sqrt(3)/2, 0])
        assert_aae(elements[0].rd, [R,     0,           0])
        assert_aae(elements[1].rd, [-R/2,  R*sqrt(3)/2, 0])
        assert_aae(elements[2].rd, [-R/2, -R*sqrt(3)/2, 0])

    def test_three_elements_forming_a_disc_about_Z_have_correct_inertia(self):
        density = 230.4
        length = 20.0
        offset = 5.0
        m = density * length  # mass of one beam
        joint = FreeJoint('joint')

        # Make 3 elements spaced by 120 deg about z axis
        for i in range(3):
            rotmat = rotations(('z', i * 2*pi/3))
            offset_vector = dot(rotmat, [offset, 0, 0])
            conn = RigidConnection('offset%d' % i, offset_vector, rotmat)
            element = _mock_rigid_uniform_beam(density, length,
                                               'element%d' % i)
            joint.add_leaf(conn)
            conn.add_leaf(element)

        system = System()
        system.add_leaf(joint)
        system.setup()

        # Calculate reduced system to get rigid body matrices
        rsys = ReducedSystem(system)

        # Expected values: perp inertia using projected lengths of beams
        Iy = m * (length**2 / 12 + (length/2 + offset)**2)
        Iperp = Iy + Iy/4 + Iy/4
        Iaxial = 3 * Iy
        expected_mass = 3 * m * eye(3)
        expected_inertia = diag([Iperp, Iperp, Iaxial])
        expected_offdiag = zeros((3, 3))

        assert_aae(rsys.M[:3, :3], expected_mass)
        assert_aae(rsys.M[3:, 3:], expected_inertia)
        assert_aae(rsys.M[3:, :3], expected_offdiag)
        assert_aae(rsys.M[:3, 3:], expected_offdiag.T)

    def test_three_elements_forming_a_disc_about_X_have_correct_inertia(self):
        density = 5
        length = 20.0
        offset = 1.25
        m = density * length  # mass of one beam

        joint = FreeJoint('joint')

        # Make 3 elements spaced by 120 deg about z axis
        for i in range(3):
            # Rotation of -pi/2 about y aligns local x axis of ModalElement
            rotmat = rotations(('x', i * 2*pi/3), ('y', -pi/2))
            offset_vector = dot(rotmat, [offset, 0, 0])  # offset // local x
            conn = RigidConnection('offset%d' % i, offset_vector, rotmat)
            element = _mock_rigid_uniform_beam(density, length,
                                               'element%d' % i)
            joint.add_leaf(conn)
            conn.add_leaf(element)

        system = System()
        system.add_leaf(joint)
        system.setup()

        # Calculate reduced system to get rigid body matrices
        rsys = ReducedSystem(system)

        # Expected values: perp inertia using projected lengths of beams
        Iy = m * (length**2 / 12 + (length/2 + offset)**2)
        Iperp = Iy + Iy/4 + Iy/4
        Iaxial = 3 * Iy
        expected_mass = 3 * m * eye(3)
        expected_inertia = diag([Iaxial, Iperp, Iperp])
        expected_offdiag = zeros((3, 3))

        assert_aae(rsys.M[:3, :3], expected_mass)
        assert_aae(rsys.M[3:, 3:], expected_inertia)
        assert_aae(rsys.M[3:, :3], expected_offdiag)
        assert_aae(rsys.M[:3, 3:], expected_offdiag.T)


# These two test cases are meant to demonstrate the differences
# between having a beam attached rigidly to ground, and being freely
# hinged. In the first case, there is no acceleration and the reaction
# forces are equal to the applied loads. In the second case, the beam
# starts to accelerate and initially the applied moment is cancelled
# by the d'Alembert inertial moment. There is however a reaction
# force remaining.

class hinged_beam_tests(unittest.TestCase):
    density = 5.0
    length = 20.0
    force = 34.2  # N/m
    hinge_torque = 0.0
    free_beam = False

    def setUp(self):
        # FE model for beam
        x = linspace(0, self.length, 20)
        fe = BeamFE(x, density=self.density, EA=0, EIy=1, EIz=0)
        fe.set_boundary_conditions('C', 'F')
        modal = fe.modal_matrices(0)
        self.beam = ModalElementFromFE('beam', modal)

        # Set loading - in negative Z direction
        load = np.zeros((len(x), 3))
        load[:, 2] = -self.force
        self.beam.loading = load

        # Hinge with axis along Y axis
        self.hinge = Hinge('hinge', [0, 1, 0])
        self.hinge.internal_torque = self.hinge_torque

        # Build system
        self.system = System()
        self.system.add_leaf(self.hinge)
        self.hinge.add_leaf(self.beam)
        self.system.setup()

        if not self.free_beam:
            # Prescribe hinge to be fixed
            self.system.prescribe(self.hinge)

        # Initial calculations
        self.recalc()

    def recalc(self):
        self.system.update_kinematics()    # Set up nodal values initially
        self.system.update_matrices()
        self.system.solve_accelerations()  # Calculate accelerations of DOFs
        self.system.update_kinematics()    # Update nodal values based on DOFs
        self.system.update_matrices()
        self.system.solve_reactions()      # Solve reactions incl d'Alembert


class Test_fixed_hinged_beam(hinged_beam_tests):
    free_beam = False

    def test_reaction_on_fixed_beam(self):
        """Reaction force should balance applied force"""
        # Applied force and moment (+ve in +Z and +Y)
        Fa = -self.force * self.length
        Ma = -Fa * self.length / 2

        # Check equilibrium. No applied forces on hinge, so both nodes
        # should be the same.
        assert_aae(self.system.joint_reactions['node-0'],
                   [0, 0, -Fa, 0, -Ma, 0])
        assert_aae(self.system.joint_reactions['ground'],
                   self.system.joint_reactions['node-0'])

        # The base of the beam should have zero acceleration:
        assert_aae(self.beam.ap, 0)

    def test_coordinate_frames(self):
        """Beam loading is in local frame, while reaction forces are
        global. So by rotating the beam, the reaction forces should
        change.
        """
        self.hinge.xstrain[0] = pi / 2
        self.recalc()

        # Applied force and moment
        Fx = -self.force * self.length   # in -X direction
        My = -Fx * self.length / 2       # in +Y direction

        # Check equilibrium. No applied forces on hinge, so both nodes
        # should be the same.
        assert_aae(self.system.joint_reactions['node-0'],
                   [-Fx, 0, 0, 0, -My, 0])
        assert_aae(self.system.joint_reactions['ground'],
                   self.system.joint_reactions['node-0'])


class Test_free_hinged_beam(hinged_beam_tests):
    free_beam = True

    def test_reaction_on_hinged_beam(self):
        """Reaction force should balance applied force"""
        # Applied force and moment (+ve in +Z and +Y)
        Fa = -self.force * self.length
        Ma = -Fa * self.length / 2

        # Acceleration
        inertia = self.density * (self.length ** 3) / 3
        ang_acc = Ma / inertia
        assert_aae(self.hinge.astrain[0], ang_acc)

        # Inertial d'Alembert force and moment
        Mi = -Ma  # moments balance initially
        Fi = self.density * self.length**2 * ang_acc / 2

        # Check equilibrium. No applied forces on hinge, so both nodes
        # should be the same.
        assert_aae(self.system.joint_reactions['node-0'],
                   [0, 0, -(Fa + Fi), 0, -(Ma + Mi), 0])
        assert_aae(self.system.joint_reactions['ground'],
                   self.system.joint_reactions['node-0'])

        # The base of the beam should have zero linear acceleration
        # but should have an angular acceleration:
        assert_aae(self.beam.ap, [0, 0, 0, 0, ang_acc, 0])


class Test_active_hinge_beam(hinged_beam_tests):
    free_beam = True
    force = 0.0
    hinge_torque = 3210.3

    def test_reaction_on_forced_beam(self):
        # Acceleration
        inertia = self.density * (self.length ** 3) / 3
        ang_acc = self.hinge_torque / inertia
        assert_aae(self.hinge.astrain[0], ang_acc)

        # Inertial d'Alembert force and moment
        Mi = -inertia * ang_acc
        Fi = self.density * self.length**2 * ang_acc / 2

        # Check equilibrium.
        assert_aae(self.system.joint_reactions['node-0'],
                   [0, 0, -Fi, 0, -Mi, 0])
        assert_aae(self.system.joint_reactions['ground'],
                   [0, 0, -Fi, 0, -Mi, 0])
