# -*- coding: utf-8 -*-
"""
Created on Wed 28 Mar 2012

Test reduction of system to independent coordinates and linearisation

@author: Rick Lupton
"""

from __future__ import division
import numpy as np
import dynamics
from dynamics import *
import dynvis

dynamics.gravity = 0


slider = PrismaticJoint('slider', [0,0,1], rotmat_y(-pi/2))
beam1 = EulerBeam('beam1', 5, 10, 1, 1, 1)
hinge = Hinge('hinge',  [0,1,0])
beam2 = EulerBeam('beam2', 7, 10, 1, 1, 1)

slider.add_leaf(beam1)
beam1.add_leaf(hinge)
hinge.add_leaf(beam2)
system = System(slider)

hinge.stiffness = 107
slider.stiffness = 54
        
# Prescribed DOF accelerations
system.prescribe(beam1.istrain, 0.0)    # rigid beam
system.prescribe(beam2.istrain, 0.0)    # rigid beam

K,C,M = system.linearise([1,1], [0,0], [0,0])

def sim_both(a,b):
    t = np.arange(0, 20, 0.05)
    ylin = solve_lin_system(M, C, K, [a,b], [0,0], t)
    system.q[system.iFreeDOF[0]] = a
    system.q[system.iFreeDOF[1]] = b
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
