# -*- coding: utf-8 -*-
"""
Misc functions
"""

from __future__ import division
from numpy import array, zeros, cos, sin

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
