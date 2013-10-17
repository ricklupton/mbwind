from numpy import zeros, array, eye, pi, dot, sqrt, c_, diag, cos, sin
from numpy import linalg
from numpy.testing import assert_array_equal, assert_array_almost_equal
from unittest import TestCase
from nose import SkipTest

from mbwind.utils import rotmat_x, rotmat_y, rotmat_z, update_skewmat
from mbwind.elements.beams import UniformBeam


class UniformBeamTestCase(TestCase):
    def test_gyroscopic_acceleration(self):
        """When the body is spinning, a torque should cause a perpendicular
        acceleration"""

        raise SkipTest

        # Set up rigid body spinning about x axis, and precessing about z axis
        precession = 0.1
        spin = 27.3
        length = 3.5
        radius = 1.2
        mass = 6.54

        # Calculate inertias
        A = mass * (3*radius**2 + 2*length**2) / 12     # perpendicular inertia
        C = mass * radius**2 / 2                        # polar inertia

        # Make a uniform beam representing a gyroscope spinning around
        # the beam's axis
        b = UniformBeam('beam', length=length, density=mass/length, Jx=C,
                        EA=1, EIy=1, EIz=1)  # these don't matter
        b.vp[3:] = [spin, 0, precession]
        update_skewmat(b.wps, b.vp[3:])
        b.calc_mass()

        # Expect moment of momentum to be [C*spin, 0, A*precession] in global
        Jp = dot(b.Rp, dot(diag([C, A, A]), b.Rp.T))
        hp = dot(Jp, b.vp[3:])
        assert_array_almost_equal(hp, [C*spin, 0, A*precession])

        # Expect the torque to be (precession * spin_speed) * (C - A)
        # about the y axis
        expected_Q2 = spin * precession * (C - A)
        Q2_at_centre = b.quad_forces[4] + b.quad_forces[10] + \
                       (length/2)*(b.quad_forces[2] - b.quad_forces[8])
        #assert_array_almost_equal(b.quad_forces, [0, 0, 0, 0, expected_Q2, 0])
        assert_array_almost_equal(Q2_at_centre, expected_Q2)
