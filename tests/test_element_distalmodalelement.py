import unittest
import numpy as np
from numpy import array, dot
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal as assert_aae)

from beamfe import BeamFE, interleave
from mbwind import skewmat, DistalModalElementFromFE
from .common import (set_random_state, constraint_consistency_check)


def _make_random_element(rdm):
    # Choose some parameters randomly
    length = rdm.uniform(1, 10)
    density = rdm.uniform(0.1, 100)
    EA = rdm.uniform(1e1, 1e8)
    EIy = rdm.uniform(1e5, 1e8)
    EIz = rdm.uniform(1e5, 1e8)

    x = np.linspace(0, length, 20)
    fe = BeamFE(x, density, EA, EIy, EIz)
    fe.set_boundary_conditions('C', 'F')
    modal = fe.modal_matrices(5)
    beam = DistalModalElementFromFE('el', modal)
    beam._params = dict(length=length,
                        density=density,
                        EA=EA, EIy=EIy, EIz=EIz)
    return beam


class TestModalElementFromFE_kinematics(unittest.TestCase):
    def setUp(self):
        self.rdm = np.random.RandomState(123456789)

    def test_distal_node_position_with_zero_strain(self):
        NTESTS = 5
        for i in range(NTESTS):
            el = _make_random_element(self.rdm)
            set_random_state(el, self.rdm, 1)
            el.xstrain[:] = 0
            el.calc_distal_pos()

            last_node_x0 = [el._params['length'], 0, 0]
            assert_aae(el.rd, el.rp + dot(el.Rp, last_node_x0))
            assert_aae(el.Rd, el.Rp)

    def test_distal_node_position(self):
        NTESTS = 5
        for i in range(NTESTS):
            el = _make_random_element(self.rdm)
            set_random_state(el, self.rdm, 1)

            last_node_x0 = [el._params['length'], 0, 0]
            Xd = dot(el.modal.shapes[-6:], el.xstrain)
            last_node_defl = Xd[0:3]
            last_node_rotn = Xd[3:6]

            el.calc_distal_pos()
            assert_aae(el.rd,
                       el.rp + dot(el.Rp, last_node_x0 + last_node_defl))
            assert_aae(el.Rd,
                       dot(el.Rp, np.eye(3) + skewmat(last_node_rotn)))

            # XXX too similar to implementation

    def test_velocity_and_acceleration_constraints_are_consistent(self):
        # Check for consistency with random conditions. Because small
        # angles are assumed, need to limit random range of strain
        # values.
        constraint_consistency_check(_make_random_element, self.rdm,
                                     dt=1e-4, ntests=5,
                                     nodal_range=0.5,
                                     strain_range=1e-1)
