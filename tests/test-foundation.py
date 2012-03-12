# -*- coding: utf-8 -*-
"""
Created on Thu Mar 01 15:01:15 2012

@author: Rick Lupton
"""

from __future__ import division
import numpy as np
from numpy import pi, zeros
from dynamics import (FreeJoint, Hinge, rotmat_x, rotmat_y,
    EulerBeam, RigidConnection, System, solve_system, NQD)
import matplotlib.pylab as plt
import dynvis

tower_height = 90.0
overhang = 0.8
r_root = 0.5
blade_length = 60.0

EIy = 1000e6
EIz = 1000e6

Ry = rotmat_y(-pi/2)
Rhb1 = rotmat_x(0 * 2*pi/3)
Rhb2 = rotmat_x(1 * 2*pi/3)
Rhb3 = rotmat_x(2 * 2*pi/3)

fstiff = zeros((6,6))
fstiff[2,2] = 1000000
foundation = FreeJoint('foundation', fstiff, post_transform=rotmat_y(-pi/2))
tower = EulerBeam('tower', tower_height, 3000, 100e6, 300e6, 300e6, 200e6)
nacelle = RigidConnection('nacelle', [0,0,overhang], rotmat_y(100*pi/180))
bearing = Hinge('bearing', [1,0,0])
hb1 = RigidConnection('hb1', np.dot(Rhb1,[0,0,1]), np.dot(Rhb1,Ry))
hb2 = RigidConnection('hb2', np.dot(Rhb2,[0,0,1]), np.dot(Rhb2,Ry))
hb3 = RigidConnection('hb3', np.dot(Rhb3,[0,0,1]), np.dot(Rhb3,Ry))
b1 = EulerBeam('b1',blade_length, 250, 1000e6, EIy, EIz, 200e6)
b2 = EulerBeam('b2',blade_length, 250, 1000e6, EIy, EIz, 200e6)
b3 = EulerBeam('b3',blade_length, 250, 1000e6, EIy, EIz, 200e6)

tower.damping = 0.05

foundation.add_leaf(tower)
tower.add_leaf(nacelle)
nacelle.add_leaf(bearing)
bearing.add_leaf(hb1)
bearing.add_leaf(hb2)
bearing.add_leaf(hb3)
hb1.add_leaf(b1)
hb2.add_leaf(b2)
hb3.add_leaf(b3)
system = System(foundation)

def thrust(xp, xd, xstrain, vp, vd, vstrain):
    f = zeros(NQD * 2)
    f[6] = -70000
    return f

def test1():
    # Initial conditions
    system.qd[bearing.istrain[0]] = -10 * pi/180

    # Prescribed DOFs
    #system.prescribe(foundation.istrain, 0.0)
    system.prescribe(tower.istrain, 0.0)
    system.prescribe(bearing.istrain, 0.0)
    system.prescribe(b1.istrain, 0.0)
    system.prescribe(b2.istrain, 0.0)
    system.prescribe(b3.istrain, 0.0)

    # Solve
    dt = 0.05
    t = np.arange(0, 10, dt)
    y = solve_system(system, t)

    return t,y

def ani_xz(t,y):
    return dynvis.anim(system, t, y, (0,2), (-30,30), (0,tower_height+blade_length+5))

def ani_yz(t,y):
    return dynvis.anim(system, t, y, (1,2), (-blade_length-2,blade_length+2), (0,tower_height+blade_length+5))

def plot_defl(t,y):
    ip  = [b+i for i in (1,5) for b in (8,14,20)]
    oop = [b+i for i in (2,4) for b in (8,14,20)]
    print ip

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(t,y[:,ip])
    #ax1.legend(('B1 tr','B1 rot','B2 tr','B2 rot','B3 tr','B3 rot'))
    ax1.set_title('In plane')
    ax1.set_xlabel('Time / s')
    ax1.set_ylabel('Strain')

    ax2 = fig.add_subplot(122)
    ax2.plot(t,y[:,oop])
    ax2.legend(('B1 tr','B2 tr','B3 tr','B1 rot','B2 rot','B3 rot'))
    ax2.set_title('Out of plane')
    ax2.set_xlabel('Time / s')
