# -*- coding: utf-8 -*-

##  IMPORTS  ##
##
##  Standard Library
from numpy import zeros, arange

##  Third Party
#
##  PyVows Testing
from pyvows import Vows, expect

##  Local Imports
from mbwind import (ModalElement, FreeJoint, System, ReducedSystem)
from mbwind.modes import ModalRepresentation


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


##  TESTS  ##


@Vows.batch
class ModalBeamElement(Vows.Context):
    class WithNoModeShapesAndTipLoading(Vows.Context):
        density = 123.5
        length = 5.4

        def topic(self):
            x = arange(0, 10.1, 1)
            modes = ModalRepresentation(x)
            beam = ModalElement('beam', modes)

            # Loading function which returns tip load in Y direction
            beam.loading = lambda el, time: 3.45 * tip_load(x, 1)
            beam.calc_mass()
            beam.calc_kinematics()
            beam.calc_external_loading()
            return beam

        def applied_forces_are_correct(self, beam):
            F = 3.45 * 1.0 / 2  # 1/2 because load is distributed over triangle
            L = 10.0 - (1.0/3)  # centre of action is 1/3 from end
            expect(beam.applied_forces).to_almost_equal([
                0, F, 0,
                0, 0, F*L,
            ])

        def reaction_forces_are_opposite_to_applied_forces(self, beam):
            system = System()
            system.add_leaf(beam)
            system.setup()
            system.update_kinematics()
            system.solve_reactions()

            expect(system.joint_reactions['ground']) \
                .to_almost_equal(-beam.applied_forces)

        # class ItsMassMatrix(Vows.Context):
        #     def topic(self, beam):
        #         return rigid_body_mass_matrix(beam)

        #     def should_have_correct_mass(self, M):
        #         m = 123.5 * 5.4
        #         expect(M[:3, :3]).to_almost_equal(m * eye(3))

        #     def should_have_inertia_of_a_rod_about_its_end(self, M):
        #         # i.e. no inertia of sections
        #         length = 5.4
        #         m = 123.5 * length
        #         Ix = 0
        #         Iy = Iz = m * (length**2 / 12 + (length / 2)**2)
        #         expect(M[3:, 3:]).to_almost_equal(diag([Ix, Iy, Iz]))

        #     def should_have_correct_mass_offset_terms(self, M):
        #         length = 5.4
        #         m = 123.5 * length
        #         expect(M[5, 1]).to_almost_equal( m * length/2)
        #         expect(M[4, 2]).to_almost_equal(-m * length/2)
        #         expect(M[1, 5]).to_almost_equal(M[5, 1])
        #         expect(M[2, 4]).to_almost_equal(M[4, 2])

        # class ReactionForces(Vows.Context):
        #     def setup(self, beam):
        #         # Apply a force to the tip
