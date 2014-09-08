from numpy import zeros, array, eye, pi, dot, sqrt, c_, diag, ones_like, arange, linspace
from numpy import linalg
from numpy.testing import assert_array_equal, assert_array_almost_equal
import numpy as np
from nose import SkipTest

from mbwind import System
from mbwind.modes import ModalRepresentation
from mbwind.elements import ModalElement
from mbwind.utils import rotmat_x, rotmat_y, rotmat_z

assert_aae = assert_array_almost_equal


def _mock_rigid_uniform_modes(density, length):
    """Return empty modal representation of 20m long rigid beam"""
    x = arange(0, length + 0.01)
    return ModalRepresentation(x, density * ones_like(x))


def rigid_body_mass_matrix(element):
    joint = FreeJoint('joint')
    system = System()
    system.add_leaf(joint)
    joint.add_leaf(element)
    system.setup()

    for el in joint.iter_leaves():
        system.prescribe(el, 0, 0)
    system.update_kinematics()
    rsys = ReducedSystem(system)
    return rsys.M


def tip_load(x, direction):
    """Triangular distributed load over last section"""
    P = zeros((len(x), 3))
    P[-1, direction] = 1
    return P


def make_a_modal_beam(length, density=None, section_inertia=None):
    x = linspace(0, length, 10)
    modes = ModalRepresentation(x, density, section_inertia=section_inertia)
    beam = ModalElement('beam', modes)
    return beam


class ModalElement_AsRigidBody_Tests:

    def setup(self):
        self.density = 230.4
        self.length = 20.0
        self.modes = _mock_rigid_uniform_modes(self.density, self.length)
        self.element = ModalElement('element', self.modes)

    def test_element_mass_when_aligned_with_global_coordinates(self):
        self.element.calc_mass()

        # Expected values: rod along x axis
        m = self.density * self.length
        Iy = m * self.length**2 / 3
        expected_mass = m * eye(3)
        expected_inertia = diag([0, Iy, Iy])
        expected_offdiag = zeros((3, 3))

        # Y accel -> positive moment about Z
        # Z accel -> negative moment about Y
        expected_offdiag[2, 1] =  m * self.length / 2
        expected_offdiag[1, 2] = -m * self.length / 2

        elmass = self.element.mass_vv
        assert_aae(elmass[:3, :3], expected_mass)
        assert_aae(elmass[3:, 3:], expected_inertia)
        assert_aae(elmass[3:, :3], expected_offdiag)
        assert_aae(elmass[:3, 3:], expected_offdiag.T)

    def test_element_mass_when_rotated_and_offset(self):
        Rp = dot(rotmat_y(pi/5), dot(rotmat_z(-pi/7), rotmat_x(3*pi/4)))
        self.element.rp[:] = [304.3, 12.3, -402.0]
        self.element.Rp[:, :] = Rp
        self.element.calc_mass()

        # Expected values: rod along x axis
        m = self.density * self.length
        Iy = m * self.length**2 / 3
        expected_mass = m * eye(3)
        expected_inertia = diag([0, Iy, Iy])
        expected_offdiag = zeros((3, 3))

        # Y accel -> positive moment about Z
        # Z accel -> negative moment about Y
        expected_offdiag[2, 1] =  m * self.length / 2
        expected_offdiag[1, 2] = -m * self.length / 2

        transform = lambda A: dot(Rp, dot(A, Rp.T))
        assert_aae(expected_mass, transform(expected_mass))

        elmass = self.element.mass_vv
        assert_aae(elmass[:3, :3], transform(expected_mass))
        assert_aae(elmass[3:, 3:], transform(expected_inertia))
        assert_aae(elmass[3:, :3], transform(expected_offdiag))
        assert_aae(elmass[:3, 3:], transform(expected_offdiag).T)


class ModalElement_WithNoModeShapesAndTipLoading_tests:
    length = 5.4
    load = 3.45

    def setup(self):
        beam = make_a_modal_beam(self.length)
        # Loading function which returns tip load in Y direction
        beam.loading = lambda el, time: self.load * tip_load(el.modes.x, 1)
        beam.calc_mass()
        beam.calc_kinematics()
        beam.calc_external_loading()
        self.beam = beam

    def test_applied_forces_are_correct(self):
        dx = self.beam.modes.x[-1] - self.beam.modes.x[-2]
        F = self.load * dx / 2    # 1/2: load distributed over triangle
        L = self.length - (dx/3)  # centre of action is 1/3 from end
        assert_aae(self.beam.applied_forces, [0, F, 0, 0, 0, F*L])


class ModalElement_SprungBeam_tests:
    length    = 9.3     # m
    mass      = 45.4    # kg
    stiffness = 8703.2  # N/rad

    def setup(self):
        x = linspace(0, self.length, 50)
        density = self.mass / self.length
        # One mode shape: linear in z direction.
        shape = linspace(0, 1, 50)
        rotation = -np.ones(50) / self.length
        shapes = c_[0 * x, 0 * x, shape].reshape((len(x), 3, 1))
        rotations = c_[0 * x, rotation, 0 * x].reshape((len(x), 3, 1))

        # Natural frequency: w^2 = k/I
        I0 = self.mass * (self.length ** 2 / 3)     # inertia about end
        freq = (self.stiffness / I0) ** 0.5
        modes = ModalRepresentation(x, density, shapes, rotations, [freq])
        self.beam = ModalElement('beam', modes)

    def test_deflection_under_load(self):
        F = 4.2  # N/m
        M = (F * self.length) * (self.length / 2)
        expected_rotation = M / self.stiffness
        expected_tip_motion = expected_rotation * self.length

        # First, calculate directly
        loading = zeros((50, 3))
        loading[:, 2] = F
        Qr, Qw, Qs = self.beam.modes.distributed_loading(loading, [0])
        actual_tip_motion = Qs / self.beam.stiffness

        assert_aae(actual_tip_motion[0], expected_tip_motion, decimal=4)

        # Now check using system equilibrium
        self.beam.loading = lambda beam, time: loading
        system = System()
        system.add_leaf(self.beam)
        system.setup()
        system.find_equilibrium()
        actual_tip_motion2 = system.q[system.elements['beam'].istrain]
        assert_aae(actual_tip_motion2[0], expected_tip_motion, decimal=4)


class ModalElement_AsARigidRod_Mass_tests:
    density = 123.5
    length = 5.4

    def setup(self):
        beam = make_a_modal_beam(self.length, self.density)
        beam.calc_mass()
        self.M = beam.mass_vv
        self.m = self.density * self.length

    def test_has_correct_mass(self):
        assert_aae(self.M[:3, :3], self.m * eye(3))

    def test_has_inertia_of_a_rod_about_its_end(self):
        # i.e. no inertia of sections
        Ix = 0
        Iy = Iz = self.m * self.length**2 / 3
        assert_aae(self.M[3:, 3:], diag([Ix, Iy, Iz]))

    def test_has_correct_mass_offset_terms(self):
        assert_aae(self.M[5, 1], +self.m * self.length/2)
        assert_aae(self.M[4, 2], -self.m * self.length/2)
        assert_aae(self.M[1, 5], self.M[5, 1])
        assert_aae(self.M[2, 4], self.M[4, 2])


class ModalElement_AsARigidCylinder_Mass_tests:
    """As a cylinder, i.e. with section inertia, unlike a rod"""
    density = 453.1
    length = 52.5
    radius = 7.4

    def setup(self):
        Jx = self.density * self.radius**2 / 2
        beam = make_a_modal_beam(self.length, self.density, Jx)
        beam.calc_mass()
        self.M = beam.mass_vv
        self.m = self.density * self.length

    def test_has_correct_mass(self):
        assert_aae(self.M[:3, :3], self.m * eye(3))

    def test_has_inertia_of_a_cylinder_about_its_end(self):
        Ix = self.m * self.radius**2 / 2
        Iy = self.m * (self.length**2 / 3 + self.radius**2 / 4)
        Iz = Iy
        assert_aae(self.M[3:, 3:], diag([Ix, Iy, Iz]))

    def test_has_correct_mass_offset_terms(self):
        assert_aae(self.M[5, 1], +self.m * self.length/2)
        assert_aae(self.M[4, 2], -self.m * self.length/2)
        assert_aae(self.M[1, 5], self.M[5, 1])
        assert_aae(self.M[2, 4], self.M[4, 2])
