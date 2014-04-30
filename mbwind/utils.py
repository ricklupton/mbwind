# -*- coding: utf-8 -*-
"""
Misc functions
"""

from __future__ import division
import numpy as np
from numpy import array, zeros, zeros_like, cos, sin, eye, dot, sqrt

eps_ijk = zeros((3, 3, 3))
eps_ijk[0, 1, 2] = eps_ijk[1, 2, 0] = eps_ijk[2, 0, 1] = 1
eps_ijk[2, 1, 0] = eps_ijk[1, 0, 2] = eps_ijk[0, 2, 1] = -1


def rotmat_x(theta):
    return array([
        [1, 0, 0],
        [0, cos(theta), -sin(theta)],
        [0, sin(theta),  cos(theta)],
    ])


def rotmat_y(theta):
    return array([
        [cos(theta), 0, sin(theta)],
        [0, 1, 0],
        [-sin(theta), 0, cos(theta)],
    ])


def rotmat_z(theta):
    return array([
        [cos(theta), -sin(theta), 0],
        [sin(theta),  cos(theta), 0],
        [0,              0,             1],
    ])


def rotations(*rotation_list):
    """Applies the rotations in turn about the updated intermediate axes.

    e.g.     rotations(('z', pi/2), ('x', pi/2))
         --> dot(rotmat_z(pi/2), rotmat_x(pi/2))

    This corresponds to the transformation matrix x = AX, where X is
    expressed in body-fixed coordinates and x is in global
    coordiates. The order of rotations is applied TO the body, where
    initially it is aligned with the global axes.
    """
    rotmat = dict(x=rotmat_x, y=rotmat_y, z=rotmat_z)
    A = eye(3)
    for direction, angle in rotation_list:
        A = dot(A, rotmat[direction](angle))
    return A


def skewmat(vec):
    """Return the matrix that calculates the cross product with `vec`.
    """
    return array([
        [0,      -vec[2],  vec[1]],
        [vec[2],  0,      -vec[0]],
        [-vec[1],  vec[0],  0],
    ])


def update_skewmat(mat, vec):
    """Update `mat` to be the skewmat of `vec`"""
    mat[0, 1] = -vec[2]
    mat[0, 2] = vec[1]
    mat[1, 0] = vec[2]
    mat[1, 2] = -vec[0]
    mat[2, 0] = -vec[1]
    mat[2, 1] = vec[0]


def qrot3(q):
    """Return the rotation matrix associated with the 3 quaternions q.

    The 0th quaternion is calculated from the 3 which are supplied.
    """
    q1, q2, q3 = q
    q0 = sqrt(1.0 - q1**2 - q2**2 - q3**2)
    assert not np.isnan(q0)
    return array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),     1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1),     1 - 2*(q1**2 + q2**2)],
    ])


def euler_param_E(q):
    return array([
        [-q[1],  q[0], -q[3],  q[2]],
        [-q[2],  q[3],  q[0], -q[1]],
        [-q[3], -q[2],  q[1],  q[0]],
    ])


def discont_trapz(y, rho, x):
    """
    Trapezium rule along first axis of y, with weights given by discontinuous
    values rho.
    """
    assert x.ndim == 1 and len(x) == y.shape[0]
    result = zeros_like(y[0])
    for i in range(y.shape[0] - 1):
        result += (x[i+1]-x[i]) * (y[i]*rho[i, 0] + y[i+1]*rho[i, 1])
    return result / 2
