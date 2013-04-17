# -*- coding: utf-8 -*-
"""
Created on Thu Mar 01 15:01:15 2012

@author: Rick Lupton
"""

from __future__ import division
import sys
if not '..' in sys.path: sys.path.append('..')

import numpy as np
import dynamics
from dynamics import *
from wind import Wind
import dynvis

thewind = Wind(20.0)

blade_length = 60.0
EIy = 1000e6
EIz = 1000e6

foundation = Hinge('foundation', [0,1,0], rotmat_y(-pi/2))
b1 = EulerBeam('b1',blade_length, 250, 1000e6, EIy, EIz, 200e6, wind=thewind)

foundation.add_leaf(b1)
system = System(foundation)

def test1():
    # Initial conditions
    
    # Prescribed DOFs
    system.prescribe(foundation.istrain, 0.0)
    system.prescribe([b1.istrain[0], b1.istrain[3]], 0.0)
    
    # Solve
    dt = 0.05
    t = np.arange(0, 20, dt)
    y = solve_system(system, t)
    
    return t,y

def ani_xz(t,y):
    return dynvis.anim(system, t, y, (0,2), (-30,30), (0,65))

def ani_yz(t,y):
    return dynvis.anim(system, t, y, (1,2), (-62,62), (0,65))

def plot_defl(t,y):
    ip  = [2,6]
    oop = [3,5]    
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(t,y[:,ip])
    #ax1.legend(('B1 tr','B1 rot','B2 tr','B2 rot','B3 tr','B3 rot'))
    ax1.set_title('In plane')
    ax1.set_xlabel('Time / s')
    ax1.set_ylabel('Strain')
    
    ax2 = fig.add_subplot(122)
    ax2.plot(t,y[:,oop])
    ax2.legend(('B1 tr','B1 rot'))
    ax2.set_title('Out of plane')
    ax2.set_xlabel('Time / s')
