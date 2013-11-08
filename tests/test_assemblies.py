from numpy import zeros, array, eye, pi, dot, sqrt, c_, diag, ones_like, arange
from numpy import linalg
from numpy.testing import assert_array_equal, assert_array_almost_equal
from spec import Spec
from nose import SkipTest

from mbwind import System, ReducedSystem
from mbwind.modes import ModalRepresentation
from mbwind.elements import ModalElement, RigidConnection, FreeJoint
from mbwind.utils import rotations

assert_aae = assert_array_almost_equal


def _mock_rigid_uniform_modes(density, length):
    """Return empty modal representation of 20m long rigid beam"""
    x = arange(0, length + 0.01)
    return ModalRepresentation(x, ones_like(x) * density)


class AssembliesOfRigidElements(Spec):

    class offset_rigid_ModalElement:

        def inertia_is_correct_when_offset_axially(self):
            density = 230.4
            length = 20.0
            offset = 5.0
            modes = _mock_rigid_uniform_modes(density, length)
            element = ModalElement('element', modes)
            conn = RigidConnection('offset', offset=[offset, 0, 0])
            joint = FreeJoint('joint')

            system = System()
            system.add_leaf(joint)
            joint.add_leaf(conn)
            conn.add_leaf(element)
            system.setup()
            system.update_kinematics()

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
            expected_offdiag[2, 1] =  m * (length/2 + offset)
            expected_offdiag[1, 2] = -m * (length/2 + offset)

            assert_aae(rsys.M[:3, :3], expected_mass)
            assert_aae(rsys.M[3:, 3:], expected_inertia)
            assert_aae(rsys.M[3:, :3], expected_offdiag)
            assert_aae(rsys.M[:3, 3:], expected_offdiag.T)


        def three_elements_forming_a_disc_have_ends_in_right_place(self):
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
            system.update_kinematics()

            r = offset
            R = offset + length
            assert_aae(elements[0].rp, [r,     0,           0])
            assert_aae(elements[1].rp, [-r/2,  r*sqrt(3)/2, 0])
            assert_aae(elements[2].rp, [-r/2, -r*sqrt(3)/2, 0])
            assert_aae(elements[0].rd, [R,     0,           0])
            assert_aae(elements[1].rd, [-R/2,  R*sqrt(3)/2, 0])
            assert_aae(elements[2].rd, [-R/2, -R*sqrt(3)/2, 0])


        def three_elements_forming_a_disc_about_Z_have_correct_inertia(self):
            density = 230.4
            length = 20.0
            offset = 5.0
            m = density * length  # mass of one beam

            modes = _mock_rigid_uniform_modes(density, length)
            joint = FreeJoint('joint')

            # Make 3 elements spaced by 120 deg about z axis
            for i in range(3):
                rotmat = rotations(('z', i * 2*pi/3))
                offset_vector = dot(rotmat, [offset, 0, 0])
                conn = RigidConnection('offset%d' % i, offset_vector, rotmat)
                element = ModalElement('element%d' % i, modes)
                joint.add_leaf(conn)
                conn.add_leaf(element)

            system = System()
            system.add_leaf(joint)
            system.setup()
            system.update_kinematics()

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


        def three_elements_forming_a_disc_about_X_have_correct_inertia(self):
            density = 5
            length = 20.0
            offset = 1.25
            m = density * length  # mass of one beam

            modes = _mock_rigid_uniform_modes(density, length)
            joint = FreeJoint('joint')

            # Make 3 elements spaced by 120 deg about z axis
            for i in range(3):
                # Rotation of -pi/2 about y aligns local x axis of ModalElement
                rotmat = rotations(('x', i * 2*pi/3), ('y', -pi/2))
                offset_vector = dot(rotmat, [offset, 0, 0])  # offset // local x
                conn = RigidConnection('offset%d' % i, offset_vector, rotmat)
                element = ModalElement('element%d' % i, modes)
                joint.add_leaf(conn)
                conn.add_leaf(element)

            system = System()
            system.add_leaf(joint)
            system.setup()
            system.update_kinematics()

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

