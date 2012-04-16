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
import matplotlib.gridspec as gridspec

import dynamics
from dynamics import System, ModalElement, solve_system
from linearisation import LinearisedSystem, ModalRepresentation
from loading import BladeLoading
import dynvis

dynamics.gravity = 0

path_damped = '/bladed/temp/parked'
path_undamped = '/bladed/temp/parked_undamped'

# Modal element using data from Bladed model
print "Loading modes from '%s'..." % path_damped
modes = ModalRepresentation.from_Bladed(path_damped+'.$pj')
print "Loading modes from '%s'..." % path_undamped
modes0 = ModalRepresentation.from_Bladed(path_undamped+'.$pj')

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
el0 = ModalElement('el0', modes0, loading)
system0 = System(el0)

# Define outputs
def outputs(system):
    tippos = system.first_element.station_positions()
    return np.r_[ tippos[16,1], tippos[32,1] ]

# Run a simluation
def simulate(system, t1=2.0, dt=0.005):
    t = np.arange(0, t1, dt)
    y = solve_system(system, t, outputs)
    return t,y

# Load Bladed data for comparison
import pybladed.data
brun = pybladed.data.BladedRun(path_damped)
bladed_defl = np.c_[
    brun.get('blade 1 x-deflection;2'),
    brun.get('blade 1 x-deflection;4'),
]
brun0 = pybladed.data.BladedRun(path_undamped)
bladed_defl0 = np.c_[
    brun0.get('blade 1 x-deflection;2'),
    brun0.get('blade 1 x-deflection;4'),
]

def plotboth(t, y, bladed):
    plt.plot(t, y[:,4:], 'k+', t, bladed[:len(t),0], 'b-',
             t, bladed[:len(t),1], 'b-')

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


def doplot():
    fig = plt.figure()
    gs = gridspec.GridSpec(3, 1)
    
    fig.suptitle('Blade deflection at midspan and tip')
    ax1 = fig.add_subplot(gs[0:2,0])
    ax1.plot(t, y[:,4], 'k', label='Mine (damped)')
    ax1.plot(t, y[:,5], 'k')
    ax1.plot(t, bladed_defl[:len(t),0], 'b:', label='Bladed (damped)')
    ax1.plot(t, bladed_defl[:len(t),1], 'b:')
    ax1.plot(t0, y0[:,4:], 'k', t0, bladed_defl0[:len(t0),0], 'b:', t0, bladed_defl0[:len(t0),1], 'b:')
    ax1.set_ylabel('Deflection / m')
    ax1.legend(frameon=False)
    ax1.set_ylim((0,0.045))
    plt.setp(ax1.get_legend().get_texts(), fontsize='small')
    
    ax2 = fig.add_subplot(gs[2,0])
    ax2.plot(wind_table[0], wind_table[1:].T)
    ax2.set_xlabel('Time / s')
    ax2.set_ylabel('Wind speed / m/s')
    ax2.legend(('x','y','z'),frameon=False,loc='lower right')
    ax2.set_ylim((0,25))
    plt.setp(ax2.get_legend().get_texts(), fontsize='small')
    
if False:
    t,y = simulate(system, 10, 0.05)
    t0,y0 = simulate(system0, 10, 0.05)

