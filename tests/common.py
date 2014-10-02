import numpy as np
from numpy import pi, array, dot
from numpy.testing import assert_array_equal, assert_array_almost_equal
from mbwind import rotations, skewmat


def random_rotation_matrix(rdm):
    return rotations(('x', rdm.uniform(-pi, pi)),
                     ('y', rdm.uniform(-pi, pi)),
                     ('z', rdm.uniform(-pi, pi)))


def _deskew(mat):
    return array([mat[2, 1], mat[0, 2], mat[1, 0]])


def test_deskew():
    rdm = np.random.RandomState(123456789)
    for i in range(10):
        vec = rdm.uniform(-10, 10, 3)
        assert_array_equal(vec, _deskew(skewmat(vec)))


def set_random_state(el, rdm, RANGE):
    el.rp[:] = rdm.uniform(-RANGE, RANGE, el.rp.shape)
    el.Rp[:, :] = random_rotation_matrix(rdm)
    el.vp[:] = rdm.uniform(-RANGE, RANGE, el.vp.shape)
    el.ap[:] = rdm.uniform(-RANGE, RANGE, el.ap.shape)
    el.xstrain[:] = rdm.uniform(-RANGE, RANGE, el.xstrain.shape)
    el.vstrain[:] = rdm.uniform(-RANGE, RANGE, el.vstrain.shape)
    el.astrain[:] = rdm.uniform(-RANGE, RANGE, el.astrain.shape)


def constraint_consistency_check(make_element_func, rdm, ntests=50, dt=1e-3):
    RANGE = 1
    EXPECTED_DECIMAL = -int(np.log10(RANGE*dt))
    if rdm is None:
        rdm = np.random  # default random number generator

    for i in range(ntests):
        # Make an element with random parameters
        el = make_element_func(rdm)

        # Initialise with random conditions
        set_random_state(el, rdm, RANGE)
        # el.xstrain[[0, 1, 2, 3, 4, 5]] = 0
        # el.vstrain[[0, 1, 2, 3, 4]] = 0
        # el.astrain[[0, 1, 2, 3, 4, 5]] = 0

        # Save a copy of the current distal node state. NB need
        # update_kinematics() to update vd & ad, not calc_kinematics()
        el.update_kinematics()
        rd1 = el.rd.copy()
        Rd1 = el.Rd.copy()
        vd1 = el.vd.copy()
        ad1 = el.ad.copy()

        # Peturb by small timestep & recalculate
        el.vp[:] += el.ap[:] * dt
        el.rp[:] += el.vp[:3] * dt
        el.Rp[:] += dot(skewmat(el.vp[3:]), el.Rp) * dt
        el.vstrain[:] += el.astrain[:] * dt
        el.xstrain[:] += el.vstrain[:] * dt + el.astrain[:] * dt**2 / 2
        el.update_kinematics()

        # Estimate velocity and acceleration from change in
        # position and velocity respectively
        vd_est = (el.rd - rd1) / dt
        Rd_dot_est = (el.Rd - Rd1) / dt
        wd_est = _deskew(dot(Rd_dot_est, el.Rd.T))
        ad_est = (el.vd - vd1) / dt

        # Check calculated velocity against approx distal position
        print(i)
        assert_array_almost_equal(vd1[:3], vd_est, EXPECTED_DECIMAL - 1,
                                  'Wrong distal translational velocity')
        assert_array_almost_equal(vd1[3:], wd_est, EXPECTED_DECIMAL - 1,
                                  'Wrong distal angular velocity')
        assert_array_almost_equal(ad1, ad_est, EXPECTED_DECIMAL - 1,
                                  'Wrong distal acceleration')
