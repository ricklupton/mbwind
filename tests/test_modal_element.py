from numpy import zeros, array, eye, pi, dot, sqrt, c_, diag, ones_like, arange
from numpy import linalg
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose import SkipTest
from spec import Spec

from mbwind.modes import ModalRepresentation
from mbwind.elements import ModalElement
from mbwind.utils import rotmat_x, rotmat_y, rotmat_z

assert_aae = assert_array_almost_equal


def _mock_rigid_uniform_modes(density, length):
    """Return empty modal representation of 20m long rigid beam"""
    x = arange(0, length + 0.01)
    return ModalRepresentation(x, density * ones_like(x))


class ModalElement_(Spec):

    class as_rigid_body:

        def setup(self):
            self.density = 230.4
            self.length = 20.0
            self.modes = _mock_rigid_uniform_modes(self.density, self.length)
            self.element = ModalElement('element', self.modes)

        def element_mass_correct_when_aligned_with_global_coordinates(self):
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

        def element_mass_correct_when_rotated_and_offset(self):
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

