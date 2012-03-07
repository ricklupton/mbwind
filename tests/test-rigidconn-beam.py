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
 
rc1 = RigidConnection('rc1', [-1.5,0.4,0.1])
beam2 = EulerBeam('beam2',3.0, 50, 0.1, 1e6)

rc1.add_leaf(beam2)
system = System(rc1)

def test1():
    beam2.damping = 0.1
    #dynamics.gravity = -9.81/10

    # Initial conditions
    #system.qd[hinge1.istrain[0]] = -2
    
    # Prescribed DOFs
    system.prescribe([beam2.istrain[0], beam2.istrain[3]], 0.0)
    
    # Solve
    dt = 0.01
    t = np.arange(0, 5, dt)
    y = solve_system(system, t)
    
    return t,y

def ani():
    return dynvis.anim(system, t, y, False)