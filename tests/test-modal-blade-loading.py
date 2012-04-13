# -*- coding: utf-8 -*-
"""
Created on Fri 13 Apr 2012

Test modal element drag loading

@author: Rick Lupton
"""

from __future__ import division

import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

import dynamics
from dynamics import System, ModalElement, solve_system
from linearisation import LinearisedSystem, ModalRepresentation
from loading import BladeLoading
import dynvis

dynamics.gravity = 0

# Modal element using data from Bladed model
print "Loading modes from 'demo_a.prj'..."
modes = ModalRepresentation.from_Bladed('demo_a_simplified.prj')

# Blade loading
wind_table = np.array([
    [0, 1, 2,  10], # time
    [0, 0, 0,  0 ], # x
    [0, 0, 20, 20], # y
    [0, 0, 0,  0 ], # z
])
loading = BladeLoading(modes.x, wind_table, None)

# Modal element
el = ModalElement('el', modes, loading)
system = System(el)

# Run a simluation
def simulate(system, t1=2.0, dt=0.005):
    t = np.arange(0, t1, dt)
    y = solve_system(system, t)
    return t,y
    
def ploty(t,y):
    ax = plt.figure().add_subplot(111)
    ax.set_color_cycle(['r','g','b','c'])
    ax.plot(t,y,'-')
    plt.legend(('Mode 1','Mode 2','Mode 3','Mode 4'))

def ani_xy(s,t,y):
    return dynvis.anim(s, t, y, (0,1), (-5,45), (-5,5), velocities=False)

def ani_xz(s,t,y):
    return dynvis.anim(s, t, y, (0,2), (-5,45), (-5,5), velocities=False)

def ani_yz(s,t,y):
    return dynvis.anim(s, t, y, (1,2), (-10,10), (-10,20), velocities=False, only_free=True)

