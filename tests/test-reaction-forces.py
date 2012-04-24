# -*- coding: utf-8 -*-
"""
Created on Tue 24 Apr 2012

Test joint reaction forces: swinging pendulum with lumped mass

@author: Rick Lupton
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import dynamics
from dynamics import (System, RigidBody, Hinge, RigidConnection, Integrator)
import dynvis

# Parameters
mass = 11.234
length = 2.54

# Build model
hinge = Hinge('hinge', [0,1,0])
link = RigidConnection('link', [length,0,0])
body = RigidBody('body', mass)

hinge.add_leaf(link)
link.add_leaf(body)
system = System(hinge)

# Custom outputs to calculate correct answer
def force_hinge_prox(s):
    theta = s.q[hinge.istrain[0]]
    thetadot = s.qd[hinge.istrain[0]]
    thetadotdot = s.qdd[hinge.istrain[0]]
    A = np.array([[ np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    Fxz = -mass * length * np.dot(A, [thetadot**2, thetadotdot])
    return [Fxz[0], 0, Fxz[1] + dynamics.gravity*mass, 0, 0, 0]

# Solver
integ = Integrator(system, ('pos','vel','acc'))
integ.add_force_output(hinge.iprox, "hinge prox")
integ.add_force_output(body.iprox, "link prox")
integ.add_force_output(body.iprox, "body prox")
integ.add_custom_output(force_hinge_prox, "correct hinge prox")

def ani_xy(s,t,y):
    return dynvis.anim(s, t, y, (0,1), (-5,45), (-5,5), velocities=False)

def ani_xz(s,t,y):
    return dynvis.anim(s, t, y, (0,2), (-5,5), (-5,5), velocities=False)

def ani_yz(s,t,y):
    return dynvis.anim(s, t, y, (1,2), (-10,10), (-10,20), velocities=False, only_free=True)

def p():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle(['r','g','b'])
    # reaction forces vs predictions
    ax.plot(t,y[:,15:18],t,y[:,21:24],'k--')
    ax.plot(t,y[:,18:21],t,y[:,24:27],'k--',alpha=0.4)
    #ax.plot(t,y[:,6:9],t,y[:,30:33],'-o',t,y[:,33:36],'--')
    #ax.plot(t,y[:,6:9],t,y[:,18:21],'k--')
    ax.legend(('x','y','z'))
    
if False:
    # Run a simluation
    t,y = integ.integrate(3.8, 0.03)
