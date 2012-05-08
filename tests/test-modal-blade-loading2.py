# -*- coding: utf-8 -*-
"""
Created on Fri 13 Apr 2012

Test modal element drag loading using a real blade model: NREL 5MW turbine

@author: Rick Lupton
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import dynamics
from dynamics import System, ModalElement, Integrator, RigidConnection, rotmat_y
from blade import Blade
from loading import BladeLoading
import dynvis

dynamics.gravity = 0

bladed_path = '/bladed/blade_nrel/parked'

# Modal element using data from Bladed model
print "Loading blade from '%s'..." % bladed_path
blade = Blade(bladed_path+'.$pj')
modes = blade.modal_rep()

# Blade loading
wind_table = np.array([
    [0, 1, 2,  10], # time
    [0, 0, 20, 20], # x
    [0, 0, 0,  0 ], # y
    [0, 0, 0,  0 ], # z
])
loading = BladeLoading(blade, wind_table, None)

# Modal element
base = RigidConnection('base', rotation=rotmat_y(-np.pi/2))
el = ModalElement('el', modes, loading)
base.add_leaf(el)
system = System(base)

integ = Integrator(system)
integ.add_output(el.output_deflections(stations=[8,16]))
integ.add_output(dynamics.CustomOutput(
    lambda s: el._get_loading()[8,:], 'loading'))
integ.add_output(dynamics.CustomOutput(
    lambda s: el.loading.relspeed, 'relspeed'))

# Load Bladed data for comparison
import pybladed.data
brun = pybladed.data.BladedRun(bladed_path)
bladed_defl = np.c_[
    brun.get('blade 1 x-deflection;1'),
    brun.get('blade 1 y-deflection;1'),
    brun.get('blade 1 x-deflection;2'),
    brun.get('blade 1 y-deflection;2'),
    brun.get('DFOUT1;1'),
    brun.get('DFIN1;1'),
    brun.get('WINDSP1;2'),
]

def ani_xy(s,t,y):
    return dynvis.anim(s, t, y, (0,1), (-5,45), (-5,5), velocities=False)

def ani_xz(s,t,y):
    return dynvis.anim(s, t, y, (0,2), (-5,45), (-5,5), velocities=False)

def ani_yz(s,t,y):
    return dynvis.anim(s, t, y, (1,2), (-10,10), (-10,20), velocities=False, only_free=True)


def doplot(t=None,y=None):
    if t is None: t=t
    if y is None: y=y
    fig = plt.figure()
    gs = gridspec.GridSpec(4, 2)
    
    fig.suptitle('Blade deflection at midspan and tip\nfor a realistic blade')

    ax = fig.add_subplot(gs[0,0])
    ax.plot(t, -y[2][:,2], 'b', label='Mine')
    ax.plot(t, bladed_defl[:,4], 'k--', label='Bladed')
    ax.set_ylabel('OOP loading (N/m)')
    #ax.legend(frameon=False)
    #ax.set_ylim((0,0.045))
    #plt.setp(ax.get_legend().get_texts(), fontsize='small')
    
    ax = fig.add_subplot(gs[1:3,0])
    ax.plot(t, -100*y[1][:,0,2], 'b', label='Mine (midspan)')
    ax.plot(t, -100*y[1][:,1,2], 'g', label='Mine (tip)')
    ax.plot(t, 100*bladed_defl[:len(t),0], 'k--', label='Bladed')
    ax.plot(t, 100*bladed_defl[:len(t),2], 'k--')
    ax.set_ylabel('OOP defl (cm)')
    ax.legend(frameon=False)
    #ax.set_ylim((0,0.045))
    plt.setp(ax.get_legend().get_texts(), fontsize='small')
    
    ax = fig.add_subplot(gs[0,1])
    ax.plot(t, y[2][:,1], 'b', label='Mine')
    ax.plot(t, bladed_defl[:,5], 'k--', label='Bladed')
    ax.set_ylabel('IP loading (N/m)')
    
    ax = fig.add_subplot(gs[1:3,1])
    ax.plot(t, 100*y[1][:,0,1], 'b', label='Mine')
    ax.plot(t, 100*y[1][:,1,1], 'g')
    ax.plot(t, 100*bladed_defl[:len(t),1], 'k--', label='Bladed')
    ax.plot(t, 100*bladed_defl[:len(t),3], 'k--')
    ax.set_ylabel('IP defl (cm)')
    #ax.legend(frameon=False)
    #ax.set_ylim((0,0.045))
    #plt.setp(ax.get_legend().get_texts(), fontsize='small')
    
    ax = fig.add_subplot(gs[3,0])
    ax.plot(wind_table[0], wind_table[1:].T)
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Wind speed (m/s)')
    ax.legend(('x','y','z'),frameon=False,loc='lower right')
    ax.set_ylim((0,25))
    plt.setp(ax.get_legend().get_texts(), fontsize='small')
    
    ax = fig.add_subplot(gs[3,1])
    ax.plot(t, y[3][:,-1])
    ax.plot(t, bladed_defl[:,6], 'k--')
    ax.set_xlabel('Time / s')
    ax.set_ylabel('Rel. wind speed (m/s)')
    ax.set_ylim((18,25))
    
if False:
    t,y = integ.integrate(10, 0.05)

