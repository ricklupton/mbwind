import numpy as np
from numpy import (zeros, array, eye, pi, dot, sqrt, c_, diag,
                   ones_like, arange, linspace)
from numpy import linalg
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose import SkipTest

from mbwind import System, ReducedSystem
from mbwind.modes import ModalRepresentation
from mbwind.elements import (ModalElement, ModalElementFromFE,
                             RigidConnection, FreeJoint, Hinge)
from mbwind.utils import rotations

from beamfe import BeamFE, interleave

assert_aae = assert_array_almost_equal


class blade_reaction_force_tests:
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

    def setup(self):
        # FE model for beam - no modes, i.e. rigid
        x = linspace(0, self.length, 20)
        density = (2 * self.mass / self.length) * (1 - x / self.length)
        fe = BeamFE(x, density=density, EA=0, EIy=1, EIz=0)
        fe.set_boundary_conditions('C', 'F')
        modal = fe.modal_matrices(0)
        self.beam = ModalElementFromFE('beam', modal)

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
