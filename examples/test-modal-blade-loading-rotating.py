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
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

import dynamics
from dynamics import System, ModalElement, Integrator, Hinge
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

integ = Integrator(system, ())
integ.add_position_output(bearing.istrain)
integ.add_custom_output(lambda s: el.modes.X(el.xstrain)[(16,32),1:3].flatten(), 'defl')
integ.add_custom_output(lambda s: el._get_loading()[16,:], 'loading')
integ.add_custom_output(lambda s: el.quad_forces, 'quad_forces')
integ.add_custom_output(lambda s: el.quad_stress, 'quad_stress')

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
    ax1.plot(t, y[:,1], 'k', label='Mine')
    ax1.plot(t, bladed_defl[:len(t),0], 'k--', label='Bladed')
    ax1.plot(t, y[:,3], 'b')
    ax1.plot(t, bladed_defl[:len(t),2], 'b--')
    ax1.set_ylabel('OOP Deflection / m')
    ax1.legend(frameon=False, loc='upper left')
    #ax1.set_ylim((0,0.045))
    plt.setp(ax1.get_legend().get_texts(), fontsize='small')

    ax2 = fig.add_subplot(gs[2:4,0])
    ax2.plot(t, y[:,2], 'k', label='Mine')
    ax2.plot(t, bladed_defl[:len(t),1], 'k--', label='Bladed')
    ax2.plot(t, y[:,4], 'b')
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
    ax3a.plot(t, y[:,6:8], '-')
    ax3a.plot(t, bladed_defl[:len(t),5:7], '--x')
    ax3a.set_ylabel('Blade loading')
    ax3a.legend(('OOP','IP'),frameon=False,loc='lower right')
    ax3a.set_xlim((0,3))
    plt.setp(ax3a.get_legend().get_texts(), fontsize='small')

    ax4 = fig.add_subplot(gs[6,0])
    ax4.plot(t, y[:,0], 'k', t, bladed_defl[:len(t),4], 'k--')
    ax4.set_xlabel('Time / s')
    ax4.set_ylabel('Rotor azimuth')

def poster_plot():
    fig = plt.figure()
    fig.set_size_inches(260.0/25.4,240.0/25.4,forward=True)
    fig.subplots_adjust(left=0.32, right=0.95, top=0.95)
    gs = gridspec.GridSpec(7, 1)
    
    plt.rc('font', size=20)
    plt.rc('lines', linewidth=2)
    labelrot = {'rotation': 'horizontal', 'multialignment': 'center'}

    ax1 = fig.add_subplot(gs[3:5,0])
    ax1.plot(t, y[:,3]*1e2, 'b', label='New code')
    ax1.plot(t, bladed_defl[:len(t),2]*1e2, 'k--', label='Bladed')
    ax1.set_ylabel('Out-of-plane\n(downwind)\ndeflection\n(cm)', labelpad=40, **labelrot)
    ax1.legend(frameon=False, loc='lower right')
    ax1.yaxis.set_major_locator(MaxNLocator(prune='lower',nbins=6))
    #ax1.set_ylim((0,0.045))
    plt.setp(ax1.get_legend().get_texts(), fontsize='small')
    plt.setp(ax1, 'xticklabels', [])

    ax2 = fig.add_subplot(gs[5:7,0])
    ax2.plot(t, y[:,4]*1e2, 'b')
    ax2.plot(t, bladed_defl[:len(t),3]*1e2, 'k--')
    ax2.set_ylabel('Deflection in\nplane of rotation\n(cm)', labelpad=20, **labelrot)
    ax2.set_xlabel('Time (s)')
    ax2.yaxis.set_major_locator(MaxNLocator(prune='lower',nbins=6))

    ax3 = fig.add_subplot(gs[0:1,0])
    ax3.plot(wind_table[0], wind_table[2],'k')
    ax3.set_ylabel('Wind speed\n(m/s)', labelpad=40, **labelrot)
    ax3.set_ylim((0,25))
    #ax3.yaxis.set_major_locator(MaxNLocator(prune='lower',nbins=3))
    ax3.set_yticks([0,20])
    plt.setp(ax3, 'xticklabels', [])
    
    ax3a = fig.add_subplot(gs[1:3,0])
    ax3a.set_color_cycle(['r','g'])
    ax3a.plot(t, y[:,6:8], '-')
    ax3a.plot(t, bladed_defl[:len(t),5:7], 'k--')
    ax3a.set_ylabel('Drag force\n(N/m)', labelpad=30, **labelrot)
    ax3a.legend(('Out-of-plane','In-plane','Bladed'),frameon=False,loc='lower right')
    ax3a.yaxis.set_major_locator(MaxNLocator(prune='lower',nbins=6))
    plt.setp(ax3a.get_legend().get_texts(), fontsize='small')
    plt.setp(ax3a, 'xticklabels', [])

    #ax4 = fig.add_subplot(gs[6,0])
    #ax4.plot(t, y[:,0], 'k', t, bladed_defl[:len(t),4], 'k--')
    #ax4.set_xlabel('Time / s')
    #ax4.set_ylabel('Rotor azimuth')
    
    fig.savefig('modal-blade-loading-rotating.pdf')

if False:
    t,y = integ.integrate(10, 0.05)

