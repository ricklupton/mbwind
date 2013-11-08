# -*- coding: utf-8 -*-

##  IMPORTS  ##
##
##  Standard Library
from numpy.testing import assert_allclose
from numpy import array, eye, diag, pi, sqrt

##  Third Party
#
##  PyVows Testing
from pyvows import Vows, expect

##  Local Imports
from mbwind import (TaperedBeam, UniformBeam, RigidConnection,
                    FreeJoint, System, ReducedSystem)


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

##  TESTS  ##

@Vows.batch
class UniformBeamElement(Vows.Context):
    def element_can_be_created(self, topic):
        beam = UniformBeam('beam',
                           length=2,
                           density=5,
                           EA=1,  # these don't matter here
                           EIy=1,
                           EIz=1)

    class AsARigidRod(Vows.Context):
        density = 123.5
        length = 5.4

        def topic(self):
            beam = UniformBeam('beam',
                               length=self.length,
                               density=self.density,
                               EA=1,  # these don't matter here
                               EIy=1,
                               EIz=1)
            beam.calc_mass()
            return beam

        class ItsMassMatrix(Vows.Context):
            def topic(self, beam):
                return rigid_body_mass_matrix(beam)

            def should_have_correct_mass(self, M):
                m = 123.5 * 5.4
                expect(M[:3, :3]).to_almost_equal(m * eye(3))

            def should_have_inertia_of_a_rod_about_its_end(self, M):
                # i.e. no inertia of sections
                length = 5.4
                m = 123.5 * length
                Ix = 0
                Iy = Iz = m * (length**2 / 12 + (length / 2)**2)
                expect(M[3:, 3:]).to_almost_equal(diag([Ix, Iy, Iz]))

            def should_have_correct_mass_offset_terms(self, M):
                length = 5.4
                m = 123.5 * length
                expect(M[5, 1]).to_almost_equal( m * length/2)
                expect(M[4, 2]).to_almost_equal(-m * length/2)
                expect(M[1, 5]).to_almost_equal(M[5, 1])
                expect(M[2, 4]).to_almost_equal(M[4, 2])


    class AsARigidCylinder(Vows.Context):
        density = 123.5
        length = 5.4
        radius = 1.2

        def topic(self):
            Jx = self.density * self.length * self.radius**2 / 2
            beam = UniformBeam('beam',
                               length=self.length,
                               density=self.density,
                               Jx=Jx,
                               EA=1,  # these don't matter here
                               EIy=1,
                               EIz=1)
            beam.calc_mass()
            return beam

        class ItsMassMatrix(Vows.Context):
            def topic(self, beam):
                return rigid_body_mass_matrix(beam)

            def should_have_correct_mass(self, M):
                m = 123.5 * 5.4
                expect(M[:3, :3]).to_almost_equal(m * eye(3))

            def should_have_inertia_of_a_cylinder_about_its_end(self, M):
                # i.e. no inertia of sections
                length = 5.4
                radius = 1.2
                m = 123.5 * length
                Ix = m * radius**2 / 2
                Iy = Iz = m * (radius**2 / 4 + length**2 / 12 + (length / 2)**2)
                expect(M[3:, 3:]).to_almost_equal(diag([Ix, Iy, Iz]))

            def should_have_correct_mass_offset_terms(self, M):
                length = 5.4
                m = 123.5 * length
                expect(M[5, 1]).to_almost_equal( m * length/2)
                expect(M[4, 2]).to_almost_equal(-m * length/2)
                expect(M[1, 5]).to_almost_equal(M[5, 1])
                expect(M[2, 4]).to_almost_equal(M[4, 2])


@Vows.batch
class TaperedBeamElement(Vows.Context):

    def element_can_be_created(self, topic):
        beam = TaperedBeam('beam',
                           length=2,
                           density=(5, 5),
                           EA=(1, 1),  # these don't matter here
                           EIy=(1, 1),
                           EIz=(1, 1))

    class WithUniformDensity(Vows.Context):
        density = 504.3
        length = 3.4

        def topic(self):
            beam = TaperedBeam('beam',
                               length=self.length,
                               density=(self.density, self.density),
                               EA=(1, 1),  # these don't matter here
                               EIy=(1, 1),
                               EIz=(1, 1))
            beam.calc_mass()
            return beam

        def should_match_UniformBeam(self, beam):
            uniform_beam = UniformBeam('uniform beam',
                                       length=self.length,
                                       density=self.density,
                                       EA=1, EIy=1, EIz=1)
            uniform_beam.calc_mass()
            expect(beam.mass_vv).to_almost_equal(uniform_beam.mass_vv)
            expect(beam.mass_ve).to_almost_equal(uniform_beam.mass_ve)
            expect(beam.mass_ee).to_almost_equal(uniform_beam.mass_ee)


    class AsARigidConicalShell(Vows.Context):
        # It's a conical shell (not a solid cone) because the shell
        # has a linear distribution of mass per unit length.

        density = 302.0
        length = 9.2
        radius = 5.2

        def topic(self):
            # Mechanics databook: kx^2 = a^2 / 2
            m = self.density * self.length / 2
            Jx = m * self.radius**2 / 2
            beam = TaperedBeam('beam',
                               length=self.length,
                               density=(self.density, 0),
                               Jx=Jx,
                               EA=(1, 1),  # these don't matter here
                               EIy=(1, 1),
                               EIz=(1, 1))
            beam.calc_mass()
            return beam

        class ItsMassMatrix(Vows.Context):
            def topic(self, beam):
                return rigid_body_mass_matrix(beam)

            def should_have_correct_mass(self, M):
                m = 302.0 * 9.2 / 2   # divide by 2 for conical shell
                expect(M[:3, :3]).to_almost_equal(m * eye(3))

            def should_have_inertia_of_a_conical_shell_about_its_end(self, M):
                length = 9.2
                radius = 5.2
                m = 302 * length / 2
                # Mechanics databook: ky^2 = a^2 / 4  +  l^2/18
                Ix = m * radius**2 / 2
                Iy = Iz = m * (radius**2 / 4 + length**2 / 18 + (length / 3)**2)
                expect(M[3:, 3:]).to_almost_equal(diag([Ix, Iy, Iz]))

            def should_have_correct_mass_offset_terms(self, M):
                length = 9.2
                m = 302 * length / 2
                expect(M[5, 1]).to_almost_equal( m * length/3)
                expect(M[4, 2]).to_almost_equal(-m * length/3)
                expect(M[1, 5]).to_almost_equal(M[5, 1])
                expect(M[2, 4]).to_almost_equal(M[4, 2])


    class AssembledIntoATaperedTower(Vows.Context):
        # This is the tower of the OC3 NREL 5MW wind turbine
        data = [map(float, line.strip().split()) for line in """10.0	4667
        17.8	4345.28
        25.5	4034.76
        33.3	3735.44
        41.0	3447.32
        48.8	3170.4
        56.6	2904.69
        64.3	2650.18
        72.1	2406.88
        79.8	2174.77
        87.6	1953.87""".splitlines()]

        expected_mass = 249718.0028
        expected_first_moment = 10838365.6
        expected_second_moment = 588576754.315919

        def topic(self):
            h, density = [array(x) for x in zip(*self.data)]
            beams = [RigidConnection('offset', [h[0], 0, 0])] + \
                    [TaperedBeam('b%d' % i,
                                 length=h[i+1] - h[i],
                                 density=(density[i], density[i+1]),
                                 EA=(1, 1),  # these don't matter here
                                 EIy=(1, 1),
                                 EIz=(1, 1))
                     for i in range(len(h) - 1)]
            for b0, b1 in zip(beams[:-1], beams[1:]):
                b0.add_leaf(b1)
            M = rigid_body_mass_matrix(beams[0])
            return M

        def mass_matrix_should_be_only_rigid_body(self, M):
            expect(M.shape).to_equal((6, 6))

        def should_have_correct_mass(self, M):
            expect(M[:3, :3]).to_almost_equal(self.expected_mass * eye(3))

        def should_have_correct_mass_offset_terms(self, M):
            expect(M[5, 1]).to_almost_equal( self.expected_first_moment, 1e-2)
            expect(M[4, 2]).to_almost_equal(-self.expected_first_moment, 1e-2)
            expect(M[1, 5]).to_almost_equal(M[5, 1])
            expect(M[2, 4]).to_almost_equal(M[4, 2])

        def should_have_correct_inertia(self, M):
            I = self.expected_second_moment
            expect(M[3:, 3:]).to_almost_equal(diag([0, I, I]), 1e-3)
