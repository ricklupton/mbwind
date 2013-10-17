from numpy import zeros, array, eye, pi, dot, sqrt, c_, diag, cos, sin, asarray, arange
from numpy.testing import assert_array_equal, assert_array_almost_equal
from unittest import TestCase

from mbwind.modes import ModesFromScratch
from mbwind.utils import rotmat_z


class ModesFromScratchTestCase(TestCase):
    def test_transform(self):
        """
        Test transformation matrix from global to local coordinates
        """

        # values not important for this test
        modes = ModesFromScratch(arange(10), 1, 1, 1, 1)

        def getY(Rp, rp, rd):
            rp, rd = asarray(rp), asarray(rd)
            return modes.transformation_to_global_coords(Rp, rp, rd)

        ###### Simple case with Rp = I ######
        Y = getY(eye(3), [0, 0, 0], [1, 0, 0])

        # Velocity of distal node directly becomes displacement if prox fixed
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        # Check now what happens if proximal node has a velocity
        assert_array_equal(dot(Y, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0])
        assert_array_equal(dot(Y, [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1])

        ###### Now with Rp = rotmat_z(45 deg) ######
        Y = getY(rotmat_z(pi/4), [0, 0, 0], [1, 0, 0])
        c, s = cos(pi/4), sin(pi/4)

        # Input velocities aligned with global axes are transformed
        assert_array_almost_equal(  # noqa
            dot(Y, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
                   [0, 0, 0, 0, 0, 0, c, -s, 0, 0, 0, 0])
        assert_array_almost_equal(  # noqa
            dot(Y, [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
                   [0, 0, 0, 0, 0, 0, c, s, 0, 0, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        assert_array_almost_equal(  # noqa
            dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 0, c, s, 0]),
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        assert_array_equal(dot(Y, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        # Check now what happens if proximal node has a velocity
        assert_array_almost_equal(  # noqa
            dot(Y, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                   [0, 0, 0, 0, 0, 0, -c, s, 0, 0, 0, 0])
        assert_array_equal(dot(Y, [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),  # noqa
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


from mbwind.blade import Blade
from mbwind.modes import cumulative_mass_moment

class ModesFromBladedTestCase(TestCase):
    def test_I0(self):
        blade = Blade('tests/parked_blade_nrel.$PJ')
        modes = blade.modal_rep()

        expected = array([
            [ 363219.27561795,       0.        ,       0.        ],
            [ 362495.05310172,       0.        ,       0.        ],
            [ 357208.24272093,       0.        ,       0.        ],
            [ 349366.38469983,       0.        ,       0.        ],
            [ 336553.78402246,       0.        ,       0.        ],
            [ 316067.5905538 ,       0.        ,       0.        ],
            [ 291731.2063883 ,       0.        ,       0.        ],
            [ 262703.95670196,       0.        ,       0.        ],
            [ 230265.3607003 ,       0.        ,       0.        ],
            [ 196122.09411738,       0.        ,       0.        ],
            [ 161185.49891596,       0.        ,       0.        ],
            [ 127304.95316146,       0.        ,       0.        ],
            [  96193.05643196,       0.        ,       0.        ],
            [  67586.67885571,       0.        ,       0.        ],
            [  42498.70894388,       0.        ,       0.        ],
            [  24492.13904613,       0.        ,       0.        ],
            [  12067.50870166,       0.        ,       0.        ],
            [   2463.53545621,       0.        ,       0.        ],
            [      0.        ,       0.        ,       0.        ]])

        assert_array_almost_equal(modes.I0, expected[0])

        I0_dist = cumulative_mass_moment(modes.X0, modes.density)
        assert_array_almost_equal(I0_dist, expected)
