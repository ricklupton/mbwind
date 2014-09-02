"""Defines the ReducedSystem, i.e. the matrices in terms of only the
free DOFs with all constraints eliminated.
"""

import numpy as np
from numpy import linalg as LA
from numpy import zeros, dot


def calc_B(system):
    dofs = system.qd.dofs.subset
    B = zeros((len(dofs), len(system.qd)), dtype=bool)
    for iz, iq in enumerate(dofs):
        B[iz, iq] = True
    B = B[:, system.iReal]
    return B


def get_constraints(system):
    """
    Return constraint jacobian \Phi_q and the vectors b and c
    """

    # Constraints relating nodes defined by elements
    # They don't vary partially with time, so b=0. c contains derivatives
    # of constraint equations themselves.
    P_nodal = system.lhs[~system.iReal, :]  # pick out constraint rows
    c_nodal = system.rhs[~system.iReal]
    b_nodal = np.zeros_like(c_nodal)

    # Add ground constraints
    P_ground = np.eye(6, P_nodal.shape[1])
    c_ground = np.zeros(6)
    b_ground = np.zeros(6)

    # Add extra user-specified constraints for prescribed accelerations etc
    # These are assumed to relate always to just one DOF.
    P_prescribed = np.zeros((len(system.prescribed_dofs), P_nodal.shape[1]))
    c_prescribed = np.zeros(P_prescribed.shape[0])
    b_prescribed = np.zeros(P_prescribed.shape[0])
    for i, (dof, acc) in enumerate(system.prescribed_dofs.items()):
        if callable(acc):
            acc = acc(system.time)
        P_prescribed[i, dof] = 1
        c_prescribed[i] = acc
        # XXX assume corresponding velocity is set correctly
        b_prescribed[i] = system.qd[dof]

    # Remove zero constraint columns from P
    return (np.r_[P_nodal, P_ground, P_prescribed][:, system.iReal],
            np.r_[b_nodal, b_ground, b_prescribed],
            np.r_[c_nodal, c_ground, c_prescribed])


def calc_projections(system):
    """Calculate the projection matrices S and R which map from indep. to
    all coords.

    qd  = R zd  +  S b
    qdd = R zdd +  S c

    """
    B = calc_B(system)
    f, n = B.shape
    P, b, c = get_constraints(system)
    SR = LA.inv(np.r_[P, B])
    S = SR[:, :n-f]
    R = SR[:, n-f:]
    Sc = dot(S, c)
    Sb = dot(S, b)
    return S, R, Sc, Sb


class ReducedSystem(object):
    def __init__(self, full_system):
        S, R, Sc, Sb = calc_projections(full_system)
        full_M = full_system.lhs[np.ix_(full_system.iReal, full_system.iReal)]
        full_Q = full_system.rhs[full_system.iReal]

        self.M = dot(R.T, dot(full_M, R))
        self.Q = dot(R.T, (full_Q - dot(full_M, Sc)))
