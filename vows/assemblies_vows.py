# -*- coding: utf-8 -*-

##  IMPORTS  ##
##
##  Standard Library
from numpy import zeros, array, eye, pi, dot, sqrt, c_, diag, ones_like, arange
from numpy import linalg

##  PyVows Testing
from pyvows import Vows, expect
import __init__

##  Local Imports
from mbwind import (ModalElement, RigidConnection,
                    FreeJoint, System, ReducedSystem, rotations)
from mbwind.modes import ModalRepresentation

##  TESTS  ##


def _mock_rigid_uniform_modes(density, length):
    """Return empty modal representation of 20m long rigid beam"""
    x = arange(0, length + 0.01)
    return ModalRepresentation(x, ones_like(x) * density)


@Vows.batch
class AssembliesOfRigidElements(Vows.Context):

    class OffsetRigidModalElement:
        density = 230.4
        length = 20.0
        offset = 5.0

        def topic(self):
            modes = _mock_rigid_uniform_modes(self.density, self.length)
            element = ModalElement('element', modes)
            conn = RigidConnection('offset', offset=[self.offset, 0, 0])
            joint = FreeJoint('joint')

            system = System(gravity=0)
            system.add_leaf(joint)
            joint.add_leaf(conn)
            conn.add_leaf(element)
            system.setup()
            system.update_kinematics()
            return system

        class SystemMassMatrix:
            def topic(self, system):
                # Calculate reduced system to get rigid body matrices
                rsys = ReducedSystem(system)
                return rsys.M

            def inertia_is_correct_when_offset_axially(self, M):
                # Expected values: rod along x axis
                density, length = self.parent.density, self.parent.length
                offset = self.parent.offset
                m = density * length
                Iy = m * (length**2 / 12 + (length/2 + offset)**2)
                expected_mass = m * eye(3)
                expected_inertia = diag([0, Iy, Iy])
                expected_offdiag = zeros((3, 3))

                # Y accel -> positive moment about Z
                # Z accel -> negative moment about Y
                expected_offdiag[2, 1] = +m * (length/2 + offset)
                expected_offdiag[1, 2] = -m * (length/2 + offset)

                expect(M[:3, :3]).to_almost_equal(expected_mass)
                expect(M[3:, 3:]).to_almost_equal(expected_inertia)
                expect(M[3:, :3]).to_almost_equal(expected_offdiag)
                expect(M[:3, 3:]).to_almost_equal(expected_offdiag.T)

        class ReactionForces:
            def topic(self, system):
                # Set applied load on modal element
                def loading(element, time):
                    P = zeros((len(element.modes.x), 3))
                    P[:, 1] = 5.4  # uniform loading in y direction
                    return P
                system.elements['element'].loading = loading
                system.update_kinematics()
                system.solve_reactions()
                return system.joint_reactions['node-0']  # start of rigid conn.

            def reaction_forces_are_correct(self, F):
                length, offset = self.parent.length, self.parent.offset
                load = 5.4 * length
                expected = zeros(6)
                expected[1] = -load  # in y-direction
                expected[5] = -load * (offset + length / 2)
                expect(F).to_almost_equal(expected)

    class ThreeRigidConnectionsFormingADisc:
        length = 20.0
        offset = 5.0

        def topic(self):
            # Make 3 elements spaced by 120 deg about z axis
            system = System()
            elements = []
            for i in range(3):
                rotmat = rotations(('z', i * 2*pi/3))
                offset_vector = dot(rotmat, [self.offset, 0, 0])
                conn = RigidConnection('offset%d' % i, offset_vector, rotmat)
                element = RigidConnection('element%d' % i, [self.length, 0, 0])
                elements.append(element)
                system.add_leaf(conn)
                conn.add_leaf(element)
            system.setup()
            system.update_kinematics()
            return elements

        def the_ends_are_in_the_right_place(self, elements):
            r = self.offset
            R = self.offset + self.length
            expect(elements[0].rp).to_almost_equal([r,     0,           0])
            expect(elements[1].rp).to_almost_equal([-r/2,  r*sqrt(3)/2, 0])
            expect(elements[2].rp).to_almost_equal([-r/2, -r*sqrt(3)/2, 0])
            expect(elements[0].rd).to_almost_equal([R,     0,           0])
            expect(elements[1].rd).to_almost_equal([-R/2,  R*sqrt(3)/2, 0])
            expect(elements[2].rd).to_almost_equal([-R/2, -R*sqrt(3)/2, 0])

    class ThreeModalElementsFormingADisc:

        class AboutZAxis:
            density = 230.4
            length = 20.0
            offset = 5.0
            m = density * length  # mass of one beam

            def topic(self):
                modes = _mock_rigid_uniform_modes(self.density, self.length)
                joint = FreeJoint('joint')

                # Make 3 elements spaced by 120 deg about z axis
                for i in range(3):
                    R = rotations(('z', i * 2*pi/3))
                    offset_vector = dot(R, [self.offset, 0, 0])
                    conn = RigidConnection('offset%d' % i, offset_vector, R)
                    element = ModalElement('element%d' % i, modes)
                    joint.add_leaf(conn)
                    conn.add_leaf(element)

                system = System()
                system.add_leaf(joint)
                system.setup()
                system.update_kinematics()
                return system

            class ItsMassMatrix:
                def topic(self, system):
                    # Calculate reduced system to get rigid body matrices
                    rsys = ReducedSystem(system)
                    return rsys.M

                def is_correct(self, M):
                    # Expected: perp inertia using projected lengths of beams
                    m, length = self.parent.m, self.parent.length
                    offset = self.parent.offset

                    Iy = m * (length**2 / 12 + (length/2 + offset)**2)
                    Iperp = Iy + Iy/4 + Iy/4
                    Iaxial = 3 * Iy
                    expected_mass = 3 * m * eye(3)
                    expected_inertia = diag([Iperp, Iperp, Iaxial])
                    expected_offdiag = zeros((3, 3))

                    expect(M[:3, :3]).to_almost_equal(expected_mass)
                    expect(M[3:, 3:]).to_almost_equal(expected_inertia)
                    expect(M[3:, :3]).to_almost_equal(expected_offdiag)
                    expect(M[:3, 3:]).to_almost_equal(expected_offdiag.T)

        class AboutXAxis:
            density = 51.3
            length = 31.0
            offset = 5.0
            m = density * length  # mass of one beam

            def topic(self):
                modes = _mock_rigid_uniform_modes(self.density, self.length)
                joint = FreeJoint('joint')

                # Set applied load on modal element
                def loading(element, time):
                    P = zeros((len(element.modes.x), 3))
                    P[:, 2] = -23.6  # uniform load in local -z direction
                    return P

                # Make 3 elements spaced by 120 deg about x axis
                for i in range(3):
                    # Rotation of -pi/2 about y aligns x axis of ModalElement
                    R = rotations(('x', i * 2*pi/3), ('y', -pi/2))
                    offset_vector = dot(R, [self.offset, 0, 0])
                    conn = RigidConnection('offset%d' % i, offset_vector, R)
                    element = ModalElement('element%d' % i, modes,
                                           loading=loading)
                    joint.add_leaf(conn)
                    conn.add_leaf(element)

                system = System()
                system.add_leaf(joint)
                system.setup()
                system.update_kinematics()
                system.solve_reactions()
                return system

            class ItsMassMatrix:
                def topic(self, system):
                    # Calculate reduced system to get rigid body matrices
                    rsys = ReducedSystem(system)
                    return rsys.M

                def is_correct(self, M):
                    # Expected: perp inertia using projected lengths of beams
                    m, length = self.parent.m, self.parent.length
                    offset = self.parent.offset

                    Iy = m * (length**2 / 12 + (length/2 + offset)**2)
                    Iperp = Iy + Iy/4 + Iy/4
                    Iaxial = 3 * Iy
                    expected_mass = 3 * m * eye(3)
                    expected_inertia = diag([Iaxial, Iperp, Iperp])
                    expected_offdiag = zeros((3, 3))

                    expect(M[:3, :3]).to_almost_equal(expected_mass)
                    expect(M[3:, 3:]).to_almost_equal(expected_inertia)
                    expect(M[3:, :3]).to_almost_equal(expected_offdiag)
                    expect(M[:3, 3:]).to_almost_equal(expected_offdiag.T)

            class ReactionForces:
                def topic(self, system):
                    F = system.joint_reactions
                    return {
                        'centre': F[system.elements['offset0'].iprox],
                        'root0': F[system.elements['element0'].iprox],
                        'root1': F[system.elements['element1'].iprox],
                        'root2': F[system.elements['element1'].iprox],
                    }

                def reaction_forces_at_root_of_elements(self, reactions):
                    length = self.parent.length
                    load = 23.6 * length

                    # Load is in local -z direction, which is global +X
                    expected = zeros(6)
                    expected[0] = -load
                    expected[4] = -load * length / 2
                    expect(reactions['root0']).to_almost_equal(expected)

                def reaction_forces_at_centre(self, reactions):
                    length, offset = self.parent.length, self.parent.offset
                    load = 23.6 * length

                    # Load is in local -z direction, which is global +X
                    expected = zeros(6)
                    expected[0] = -3 * load  # moments cancel out
                    expect(reactions['centre']).to_almost_equal(expected)
