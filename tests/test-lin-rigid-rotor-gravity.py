# -*- coding: utf-8 -*-
"""
Created on Wed 28 Mar 2012

Test linearisation of a flexible 3-bladed rotor which can bounce vertically

@author: Rick Lupton
"""

from __future__ import division
import numpy as np
import dynamics
from dynamics import *
import dynvis

Ry = rotmat_y(-pi/2)
Rhb1 = rotmat_z(0 * 2*pi/3)
Rhb2 = rotmat_z(1 * 2*pi/3)
Rhb3 = rotmat_z(2 * 2*pi/3)

slider = PrismaticJoint('slider', [0,0,1], Ry)
beam = EulerBeam('beam', 5, 10, 1, 1, 1)
hinge = Hinge('hinge',  [0,0,1])
hb1 = RigidConnection('hb1', np.dot(Rhb1,[1,0,0]), Rhb1)
hb2 = RigidConnection('hb2', np.dot(Rhb2,[1,0,0]), Rhb2)
hb3 = RigidConnection('hb3', np.dot(Rhb3,[1,0,0]), Rhb3)
b1 = EulerBeam('b1', 7, 10, 1, 1, 1)
b2 = EulerBeam('b2', 7, 10, 1, 1, 1)
b3 = EulerBeam('b3', 7, 10, 1, 1, 1)

slider.add_leaf(beam)
beam.add_leaf(hinge)
hinge.add_leaf(hb1)
hinge.add_leaf(hb2)
hinge.add_leaf(hb3)
hb1.add_leaf(b1)
hb2.add_leaf(b2)
hb3.add_leaf(b3)
system = System(slider)

slider.stiffness = 54
        
# Prescribed DOF accelerations
system.prescribe(beam.istrain, 0.0)    # rigid beam
system.prescribe(b1.istrain, 0.0)    # rigid beam
system.prescribe(b2.istrain, 0.0)    # rigid beam
system.prescribe(b3.istrain, 0.0)    # rigid beam

z0 = -(50+3*70)*dynamics.gravity / slider.stiffness

linsys = system.linearise([z0,0], [0,0], [0,0])
W,Vr = scipy.linalg.eig(linsys.K,linsys.M)

def sim_both(a,b,a1,b1):
    t = np.arange(0, 20, 0.05)
    ylin = linsys.integrate(t, [a,b], [a1,b1])
    system.q [system.iFreeDOF[0]] = a
    system.q [system.iFreeDOF[1]] = b
    system.qd[system.iFreeDOF[0]] = a1
    system.qd[system.iFreeDOF[1]] = b1
    y = solve_system(system,t)
    
    # pick out 2 interesting strains
    y = y[:,[0,7]]
    ylin = ylin[:,:2]
    ax = plt.figure().add_subplot(111)
    ax.set_color_cycle(['b','r'])
    ax.plot(t,ylin,':',t,y,'--')
    return t,y,ylin
    
def simulate(system, t1, dt=0.05):
    t = np.arange(0, t1, dt)
    y = solve_system(system, t)
    return t,y

def ani_xy(s,t,y):
    return dynvis.anim(s, t, y, (0,1), (-10,10), (-10,10), velocities=False, only_free=True)

def ani_xz(s,t,y):
    return dynvis.anim(s, t, y, (0,2), (-10,10), (-10,20), velocities=False, only_free=True)

def ani_yz(s,t,y):
    return dynvis.anim(s, t, y, (1,2), (-10,10), (-10,20), velocities=False, only_free=True)

