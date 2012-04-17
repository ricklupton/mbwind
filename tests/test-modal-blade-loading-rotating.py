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
from dynamics import System, ModalElement, solve_system, Hinge
from linearisation import ModalRepresentation
from loading import BladeLoading
import dynvis

dynamics.gravity = 0

bladedpath = '/bladed/temp/simple_10rpm_nowakedyn_noinflow'

# Modal element using data from Bladed model
print "Loading modes from '%s'..." % bladedpath
modes = ModalRepresentation.from_Bladed(bladedpath+'.$pj')

# Blade loading
wind_table = np.array([
    [0, 1, 2,  10], # time
    [0, 0, 0,  0 ], # x
    [0, 0, 20, 20], # y
    [0, 0, 0,  0 ], # z
])
loading = BladeLoading(modes.x, wind_table, None)

rotorspeed = 10 * np.pi/30 # 10 rpm

# Modal element
bearing = Hinge('bearing', [0,1,0])
el = ModalElement('el', modes, loading)
bearing.add_leaf(el)
system = System(bearing)

# Prescribe rotation speed
system.prescribe(bearing.istrain, 0.0)
system.qd[bearing.istrain] = rotorspeed

system.find_equilibrium()

# Define outputs
def outputs(system):
    defl = el.modes.X(el.xstrain)
    loading = el._get_loading()
    return np.r_[ defl[16,1], defl[16,2], defl[32,1], defl[32,2],
                  loading[16,:], el.quad_forces, el.quad_stress ]

# Run a simluation
def simulate(system, t1=2.0, dt=0.005):
    t = np.arange(0, t1, dt)
    y = solve_system(system, t, outputs)
    return t,y

# Load Bladed data for comparison
import pybladed.data
brun = pybladed.data.BladedRun(bladedpath)
bladed_defl = np.c_[
    brun.get('blade 1 x-deflection;2'),
    brun.get('blade 1 y-deflection;2'),
    brun.get('blade 1 x-deflection;4'),
    brun.get('blade 1 y-deflection;4'),
    brun.get('rotor azimuth'),
    brun.get('DFOUT1;2'),
    brun.get('DFIN1;2'),
]

def ani_xy(s,t,y):
    return dynvis.anim(s, t, y, (0,1), (-5,45), (-5,5), velocities=False)

def ani_xz(s,t,y):
    return dynvis.anim(s, t, y, (0,2), (-5,45), (-5,5), velocities=False)

def ani_yz(s,t,y):
    return dynvis.anim(s, t, y, (1,2), (-10,10), (-10,20), velocities=False, only_free=True)

def doplot():
    fig = plt.figure()
    fig.set_size_inches(10,10,forward=True)
    gs = gridspec.GridSpec(7, 1)
    
    fig.suptitle('Rotating blade with wind drag')
    ax1 = fig.add_subplot(gs[0:2,0])
    ax1.plot(t, y[:,5], 'k', label='Mine')
    ax1.plot(t, bladed_defl[:len(t),0], 'k--', label='Bladed')
    ax1.plot(t, y[:,7], 'b')
    ax1.plot(t, bladed_defl[:len(t),2], 'b--')
    ax1.set_ylabel('OOP Deflection / m')
    ax1.legend(frameon=False, loc='upper left')
    #ax1.set_ylim((0,0.045))
    plt.setp(ax1.get_legend().get_texts(), fontsize='small')

    ax2 = fig.add_subplot(gs[2:4,0])
    ax2.plot(t, y[:,6], 'k', label='Mine')
    ax2.plot(t, bladed_defl[:len(t),1], 'k--', label='Bladed')
    ax2.plot(t, y[:,8], 'b')
    ax2.plot(t, bladed_defl[:len(t),3], 'b--')
    ax2.set_ylabel('IP Deflection / m')
    
    ax3 = fig.add_subplot(gs[4:6,0])
    ax3.set_color_cycle(['r','g','b'])
    ax3.plot(wind_table[0], wind_table[2],'k:')
    ax3.set_ylabel('Wind speed / m/s')
    ax3.set_ylim((0,25))
    ax3.set_xlim((0,3))
    ax3a = ax3.twinx()
    ax3a.set_color_cycle(['r','g'])
    ax3a.plot(t, y[:,10:12], '-')
    ax3a.plot(t, bladed_defl[:len(t),5:7], '--x')
    ax3a.set_ylabel('Blade loading')
    ax3a.legend(('OOP','IP'),frameon=False,loc='lower right')
    ax3a.set_xlim((0,3))
    plt.setp(ax3a.get_legend().get_texts(), fontsize='small')
    
    ax4 = fig.add_subplot(gs[6,0])
    ax4.plot(t, y[:,0], 'k', t, bladed_defl[:len(t),4], 'k--')
    ax4.set_xlabel('Time / s')
    ax4.set_ylabel('Rotor azimuth')
    
if False:
    t,y = simulate(system, 10, 0.05)

