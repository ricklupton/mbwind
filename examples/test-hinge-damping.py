# -*- coding: utf-8 -*-
"""
Created on Tue 14 August 2012

Test damping in hinges

@author: Rick Lupton
"""

from __future__ import division
import sys
import numpy as np
import matplotlib.pyplot as plt

import dynamics
from dynamics import (System, RigidBody, Hinge, RigidConnection, Integrator)
import dynvis

dynamics.OPT_GRAVITY = False

# Parameters
mass = 10.0
length = 2.5
natfreq = 1 # Hz
damping_factor = 0.05

I2 = mass * length**2
k = (2*np.pi*natfreq)**2 * I2
c = 2 * damping_factor * I2 * (2*np.pi*natfreq)

# Build model
hinge = Hinge('hinge', [0,0,1])
hinge.stiffness = k
hinge.damping = c
link = RigidConnection('link', [length, 0, 0])
body = RigidBody('body', mass)

hinge.add_leaf(link)
link.add_leaf(body)
system = System(hinge)

# Solver
integ = Integrator(system, ('pos','vel','acc'))
system.q[hinge.istrain][0] = 0.3

##########

def ani_xy(s,t,y):
    return dynvis.anim(s, t, y, (0,1), (-5,45), (-5,5), velocities=False)

def ani_xz(s,t,y):
    return dynvis.anim(s, t, y, (0,2), (-5,5), (-5,5), velocities=False)

def ani_yz(s,t,y):
    return dynvis.anim(s, t, y, (1,2), (-10,10), (-10,20), velocities=False, only_free=True)

def p(t, y):
    fig = plt.figure()

    # Hing angle
    ax = fig.add_subplot(111)
    #ax.set_color_cycle(['r','g','b'])
    ax.plot(t, np.unwrap(y[0][:,0]))
    ax.set_title('Hinge angle')

#############

if len(sys.argv) > 1:
    # Run a simluation
    t,y = integ.integrate(5, 0.01)
