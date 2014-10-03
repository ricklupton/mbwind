import unittest
import numpy as np
from numpy import array, dot
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal as assert_aae)

from beamfe import BeamFE, interleave
from mbwind import ModalElementFromFE


class TestModalElementFromFE_mode_frequency(unittest.TestCase):
    def test_first_mode_frequency(self):
        # From Reddy1993, p. 160
        x = np.linspace(0, 1, 16)

        # Using the z axis as the transverse direction gives the same
        # sign convention as Reddy uses in 2D, namely that rotations
        # are positive clockwise.
        fe = BeamFE(x, density=1, EA=0, EIy=1, EIz=0)
        fe.set_boundary_conditions('C', 'F')
        fe.set_dofs([False, False, True, False, True, False])
        modal = fe.modal_matrices()
        element = ModalElementFromFE('elem', modal)

        Mmodal = element.mass_ee
        Kmodal = np.diag(element.stiffness)
        w = np.sqrt(np.diag(Kmodal) / np.diag(Mmodal))
        assert_aae(w[0], 3.5160, decimal=4)


class TestModalElementFromFE_distributed_loading(unittest.TestCase):
    def test_static_deflection(self):
        x = array([0.0, 4.0, 10.0])
        EI = 144.0
        # Using the z axis as the transverse direction gives the same
        # sign convention as Reddy uses in 2D, namely that rotations
        # are positive clockwise.
        fe = BeamFE(x, density=10, EA=0, EIy=EI, EIz=0)
        fe.set_boundary_conditions('C', 'F')
        fe.set_dofs([False, False, True, False, True, False])
        modal = fe.modal_matrices()
        element = ModalElementFromFE('elem', modal)

        # Distributed load, linearly interpolated
        load = np.zeros((3, 3))
        load[-1, 2] = -100        # Load in z direction at tip
        element.apply_distributed_loading(load)
        defl = -element.applied_stress / element.stiffness

        # Check against directly calculating static deflection from FE
        Q = fe.distribute_load(interleave(load, 6))
        defl_fe, reactions_fe = fe.static_deflection(Q)
        assert_aae(dot(element.modal.shapes, defl), defl_fe, decimal=2)

    def test_rigid_element_reference_loading(self):
        # Make an element with no modes (rigid)
        fe = BeamFE(np.linspace(0, 1, 11), density=1, EA=0, EIy=1, EIz=0)
        fe.set_boundary_conditions('C', 'F')
        fe.set_dofs([False, False, True, False, True, False])
        modal = fe.modal_matrices(0)
        element = ModalElementFromFE('elem', modal)

        # Distributed load, linearly interpolated
        load = np.zeros((11, 3))
        load[:, 2] = -100        # Uniform load in z direction
        element.apply_distributed_loading(load)
        assert_aae(element.applied_stress, [])
        assert_aae(element.applied_forces[0:6], [0, 0, -100 * 1, 0, 50, 0])
        assert_aae(element.applied_forces[6:12], 0)

        # Distributed load, triangle at tip
        load = np.zeros((11, 3))
        load[-1, 2] = -100             # tapered load in z direction
        element.applied_forces[:] = 0  # reset
        element.apply_distributed_loading(load)
        F = -100 * 0.1 / 2
        assert_aae(element.applied_forces[0:6],
                   [0, 0, F, 0, -F * (0.9 + 0.1 * 2 / 3), 0])
        assert_aae(element.applied_forces[6:12], 0)
