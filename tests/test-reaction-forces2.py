# -*- coding: utf-8 -*-
"""
Created on Tue 24 Apr 2012

Test joint reaction forces: prescribed motion of base of link with mass at top

@author: Rick Lupton
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import dynamics
from dynamics import (System, RigidBody, PrismaticJoint, RigidConnection, Integrator)
import dynvis

# Parameters
mass = 11.234
length = 2.54

# Build model
slider = PrismaticJoint('slider', [1,0,0])
link = RigidConnection('link', [0,0,length])
body = RigidBody('body', mass)

slider.add_leaf(link)
link.add_leaf(body)
system = System(slider)

# Prescribe motion -- sinusoidal acceleration
motion_frequency = 1 # Hz
motion_amplitude = 2.3 # m
def prescribed_acceleration(t):
    w = 2*np.pi*motion_frequency
    # x =  motion_amplitude * np.cos(w*t)
    # v = -motion_amplitude * np.sin(w*t) * w
    # a = -motion_amplitude * np.cos(w*t) * w**2
    return -w**2 * motion_amplitude*np.cos(w*t)
    
system.prescribe(slider.istrain[0], acc=prescribed_acceleration)
system.q[slider.istrain[0]] = motion_amplitude

# Custom outputs to calculate correct answer
def force_body_prox(s):
    a = prescribed_acceleration(s.time)
    Fx = mass * a
    Fz = mass * dynamics.gravity
    return [Fx, 0, Fz, 0, 0, 0]

def force_link_prox(s):
    a = prescribed_acceleration(s.time)
    Fx = mass * a
    Fz = mass * dynamics.gravity
    My = length * Fx
    return [Fx, 0, Fz, 0, My, 0]

def force_slider_prox(s):
    a = prescribed_acceleration(s.time)
    x = -a / (2*np.pi*motion_frequency)**2
    Fx = mass * a
    Fz = mass * dynamics.gravity
    My = length*Fx - x*Fz
    return [Fx, 0, Fz, 0, My, 0]
  
# Solver
integ = Integrator(system, ('pos','vel','acc'))
integ.add_force_output(slider.iprox, "slider prox")
integ.add_force_output(link.iprox, "link prox")
integ.add_force_output(body.iprox, "body prox")
integ.add_custom_output(force_slider_prox, "correct slider prox")
integ.add_custom_output(force_link_prox, "correct link prox")
integ.add_custom_output(force_body_prox, "correct body prox")

def ani_xy(s,t,y):
    return dynvis.anim(s, t, y, (0,1), (-5,45), (-5,5), velocities=False)

def ani_xz(s,t,y):
    return dynvis.anim(s, t, y, (0,2), (-5,5), (-5,5), velocities=False)

def ani_yz(s,t,y):
    return dynvis.anim(s, t, y, (1,2), (-10,10), (-10,20), velocities=False, only_free=True)

def p():
    fig = plt.figure()

    # reaction forces vs predictions - base of slider        
    ax = fig.add_subplot(321)
    ax.set_color_cycle(['r','g','b'])
    ax.plot(t,y[:,3:6],t,y[:,21:24],'k--')
    ax.set_title('Slider prox')

    # reaction forces vs predictions - base of slider moments
    ax = fig.add_subplot(322)
    ax.set_color_cycle(['r','g','b'])
    ax.plot(t,y[:,6:9],t,y[:,24:27],'k--')
    ax.set_title('- moments')

    # reaction forces vs predictions - base of link        
    ax = fig.add_subplot(323)
    ax.set_color_cycle(['r','g','b'])
    ax.plot(t,y[:,9:12],t,y[:,27:30],'k--')
    ax.set_title('Link prox - forces')

    # reaction forces vs predictions - base of link moments
    ax = fig.add_subplot(324)
    ax.set_color_cycle(['r','g','b'])
    ax.plot(t,y[:,12:15],t,y[:,30:33],'k--')
    ax.set_title('- moments')
    
    # reaction forces vs predictions - base of body        
    ax = fig.add_subplot(325)
    ax.set_color_cycle(['r','g','b'])
    ax.plot(t,y[:,15:18],t,y[:,33:36],'k--')
    ax.set_title('Body prox - forces')

    # reaction forces vs predictions - base of body moments
    ax = fig.add_subplot(326)
    ax.set_color_cycle(['r','g','b'])
    ax.plot(t,y[:,18:21],t,y[:,36:39],'k--')
    ax.set_title('- moments')
    
    ax.legend(('x','y','z'))
    
if False:
    # Run a simluation
    t,y = integ.integrate(3.8, 0.03)
