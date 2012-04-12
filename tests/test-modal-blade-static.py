# -*- coding: utf-8 -*-
"""
Created on Fri 6 Apr 2012

Test modal element using blade mode shapes from Bladed: static model

@author: Rick Lupton
"""

from __future__ import division

import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

import dynamics
from dynamics import System, ModalElement, solve_system
from linearisation import LinearisedSystem, ModalRepresentation
import dynvis

dynamics.gravity = 0

# Modal element using data from Bladed model
print "Loading modes from 'demo_a.prj'..."
modes = ModalRepresentation.from_Bladed('demo_a_simplified.prj')
el = ModalElement('el', modes)
system = System(el)

# Linearise system and find modes - should match with original
print "Linearising..."
linsys = LinearisedSystem(system)
w,v = eig(linsys.K, linsys.M)
order = np.argsort(w)
w = np.sqrt(np.real(w[order]))
f = w/2/np.pi
v = v[:,order]

# Check that modes come out matching what went in
assert np.allclose(modes.freqs, w), "Linearised freqs should match"
ventries = np.nonzero(abs(v)>1e-2)[0]
assert len(ventries) == len(w) and (ventries == np.arange(4)).all(), \
    "Modes should be orthogonal"

# Run a simluation
def sim_both(q0=None, qd0=None):
    if q0  is None: q0  = np.zeros_like(w)
    if qd0 is None: qd0 = np.zeros_like(w)
    
    t = np.arange(0, 2, 0.005)
    ylin = linsys.integrate(t, q0, qd0)
    system.q [:] = 0.0
    system.qd[:] = 0.0
    system.q [system.iFreeDOF] = q0
    system.qd[system.iFreeDOF] = qd0
    y = solve_system(system,t)

    ylin = ylin[:,:len(q0)]
    ax = plt.figure().add_subplot(111)
    ax.set_color_cycle(['r','g','b','c'])
    ax.plot(t,ylin,':',t,y,'--')
    plt.legend(('Mode 1','Mode 2','Mode 3','Mode 4'))
    return t,y,ylin
    
#t,y,ylin = sim_both([0.2,0.3,0.5,0])

def simulate(system, t1, dt=0.05):
    t = np.arange(0, t1, dt)
    y = solve_system(system, t)
    return t,y

def ani_xy(s,t,y):
    return dynvis.anim(s, t, y, (0,1), (-5,65), (-10,10), velocities=False)

def ani_xz(s,t,y):
    return dynvis.anim(s, t, y, (0,2), (-5,65), (-10,20), velocities=False)

def ani_yz(s,t,y):
    return dynvis.anim(s, t, y, (1,2), (-10,10), (-10,20), velocities=False, only_free=True)

