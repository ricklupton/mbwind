# -*- coding: utf-8 -*-
"""
Created on Tue 20 Mar 2012

Model a pendulum hinged freely to the edge of a disk which spins at constant speed.

@author: Rick Lupton
"""

from __future__ import division
import numpy as np
import dynamics
from dynamics import *
import dynvis

dynamics.gravity = 0

rdisc = 1.0
lpend = 1.0

bearing = Hinge('bearing', [0,0,1])
disc = RigidConnection('disc', [rdisc,0,0])
hinge_ip  = Hinge('hinge_ip',  [0,0,1])
hinge_oop = Hinge('hinge_oop', [0,1,0])
beam = EulerBeam('beam', lpend, 250, 1, 1, 1)

bearing.add_leaf(disc)
disc.add_leaf(hinge_ip)
hinge_ip.add_leaf(hinge_oop)
hinge_oop.add_leaf(beam)
system = System(bearing)

# Prescribed DOF accelerations and initial conditions
system.prescribe(beam.istrain, 0.0)        # rigid beam
system.prescribe(bearing.istrain, 0.0)     # constant rotational speed
system.qd[bearing.istrain[0]] = 10 * pi/180 # initial rotational speed
#system.q[hinge_oop.istrain[0]] = 15 * pi/180 # initial OOP angle
system.q[hinge_ip.istrain[0]] = 15 * pi/180 # initial IP angle

#hinge_ip.stiffness = 1e3
#hinge_oop.stiffness = 1e3

def test1():
    # Solve
    dt = 0.1
    t = np.arange(0,40, dt)
    y = solve_system(system, t, outputs=lambda s: [s.rhs[i] for i in beam.iprox])
    
    return t,y

def ani_xy(t,y):
    return dynvis.anim(system, t, y, (0,1), (-3,3), (-3,3), velocities=False)

def ani_xz(t,y):
    return dynvis.anim(system, t, y, (0,2), (-3,3), (-3,3), velocities=False)

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
