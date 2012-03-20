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

class RotatingBeam(object):
    def __init__(self, r, l, mass):
        self.r = r
        self.l = l
        
        self.bearing = Hinge('bearing', [0,0,1])
        self.disc = RigidConnection('disc', [r,0,0])
        self.hinge_ip  = Hinge('hinge_ip',  [0,0,1])
        self.hinge_oop = Hinge('hinge_oop', [0,1,0])
        self.beam = EulerBeam('beam', l, mass/l, 1, 1, 1)
        
        self.bearing.add_leaf(self.disc)
        self.disc.add_leaf(self.hinge_ip)
        self.hinge_ip.add_leaf(self.hinge_oop)
        self.hinge_oop.add_leaf(self.beam)
        self.system = System(self.bearing)

        #hinge_ip.stiffness = 1e3
        #hinge_oop.stiffness = 1e3

        # Prescribed DOF accelerations
        self.system.prescribe(self.beam.istrain, 0.0)    # rigid beam
        self.system.prescribe(self.bearing.istrain, 0.0) # constant rotational speed

    def init(self, rotspeed, ip0, oop0):
        # reset
        self.system.q [:] = 0.0
        self.system.qd[:] = 0.0        
        
        # initial conditions
        self.system.qd[self.bearing.istrain[0]] = rotspeed # initial rotational speed
        self.system.qd[self.hinge_oop.istrain[0]] = oop0    # initial OOP angle
        self.system.qd[self.hinge_ip .istrain[0]] = ip0     # initial IP angle

    def simulate(self, t1, dt=0.05):
        t = np.arange(0, t1, dt)
        y = solve_system(self.system, t)
        return t,y

def ani_xy(s,t,y):
    return dynvis.anim(s, t, y, (0,1), (-3,3), (-3,3), velocities=False)

def ani_xz(s,t,y):
    return dynvis.anim(s, t, y, (0,2), (-3,3), (-3,3), velocities=False)


rdisc = 1.0
lpend = 0.2
mass = 100

def test(r,l,m,speeds):
    PHI = 1
    PSI = 2
    
    rb = RotatingBeam(r, l, m)
    # results
    ip = []
    oop = []
    for Omega in speeds:
        rotor_period = 2*pi/Omega

        # in-plane
        rb.init(Omega, 0.1, 0.0)
        t,y = rb.simulate(rotor_period)
        t_half = t[np.nonzero(y[:,PHI]<0)[0][0]]
        ip.append(pi/t_half)
        
        # out-of-plane
        rb.init(Omega, 0.0, 0.1)
        t,y = rb.simulate(rotor_period)
        t_half = t[np.nonzero(y[:,PSI]<0)[0][0]]
        oop.append(pi/t_half)
        
    return ip,oop

speeds = [0.1,0.2,0.5,1.0]
ip,oop = test(rdisc,lpend,mass,speeds)

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
