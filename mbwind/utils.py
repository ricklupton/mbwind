# -*- coding: utf-8 -*-
"""
Misc functions
"""

from __future__ import division
from numpy import array, zeros, zeros_like, cos, sin, eye, dot

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
         --> dot(rotmat_x(pi/2), rotmat_z(pi/2))
    """
    rotmat = dict(x=rotmat_x, y=rotmat_y, z=rotmat_z)
    mats = [rotmat[direction](angle) for direction, angle in rotation_list]
    return reduce(dot, mats, eye(3))


def skewmat(vec):
    return array([
        [ 0,      -vec[2],  vec[1]],
        [ vec[2],  0,      -vec[0]],
        [-vec[1],  vec[0],  0     ],
    ])


def update_skewmat(mat, vec):
    mat[0, 1] = -vec[2]
    mat[0, 2] =  vec[1]
    mat[1, 0] =  vec[2]
    mat[1, 2] = -vec[0]
    mat[2, 0] = -vec[1]
    mat[2, 1] =  vec[0]


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
