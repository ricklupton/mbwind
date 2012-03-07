# -*- coding: utf-8 -*-
"""
Created on Thu Mar 01 15:01:15 2012

@author: Rick Lupton
"""

from __future__ import division
import numpy as np
import dynamics
from dynamics import *
import dynvis
 
hinge1 = Hinge('hinge1', [0,1,0])
beam2 = EulerBeam('beam2',3.0, 50, 0.1, 1e6)

hinge1.add_leaf(beam2)
system = System(hinge1)

def test1():
    beam2.damping = 0.2
    hinge1.damping = 0.2
    dynamics.gravity = 9.81

    # Initial conditions
    #system.qd[hinge1.istrain[0]] = -2
    
    # Prescribed DOFs
    system.prescribe([beam2.istrain[0], beam2.istrain[3]], 0.0)
    
    # Solve
    dt = 0.01
    t = np.arange(0, 21, dt)
    y = solve_system(system, t)
    
    return t,y

def test2():
    beam2.damping = 0.0
    hinge1.damping = 0.3
    hinge1.stiffness = 3.0
    dynamics.gravity = 0

    # Initial conditions
    system.qd[hinge1.istrain[0]] = -1
    
    # Prescribed DOFs
    system.prescribe(beam2.istrain, 0.0)
    
    # Solve
    dt = 0.01
    t = np.arange(0, 10, dt)
    y = solve_system(system, t)
    
    return t,y
    
def ani():
    return dynvis.anim(system, t, y, False)