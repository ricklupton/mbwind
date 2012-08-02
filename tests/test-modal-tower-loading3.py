# -*- coding: utf-8 -*-
"""
Created on Fri 13 Apr 2012

Test modal element drag loading using a real blade model: NREL 5MW turbine

@author: Rick Lupton
"""

from __future__ import division

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import dynamics
from dynamics import System, ModalElement, Integrator, RigidConnection, rotmat_y, \
    RigidBody
from blade import Tower
#from loading import PointLoading
import dynvis
from linearisation import LinearisedSystem

dynamics.OPT_GRAVITY = False

modes_path = '/media/data/bladed/nrel_tower/point_load/tower_7modes_only'
bladed_path = '/media/data/bladed/nrel_tower/point_load/flextower_rigidblades_7modes'

# Modal element using data from Bladed model
print "Loading tower from '%s'..." % modes_path
#tower = Tower(modes_path+'.$pj')
tower = Tower(bladed_path+'.$PJ')
modes = tower.modal_rep(False)

# Get combined tower frequencies
#tower2 = Tower(bladed_path+'.$pj')
#modes2 = tower2.modal_rep()

isolated_freqs = 2*np.pi*np.array([1.28, 1.28, 20, 5.75, 5.75, 5.29, 5.29])
combined_freqs = modes.freqs
modes.freqs = isolated_freqs # to get correct stiffness

# Load Bladed data for comparison
import pybladed.data
brun = pybladed.data.BladedRun(bladed_path)
bladed_defl = np.c_[
    brun.get('Nacelle fore-aft displacement'),
    brun.get('Nacelle nod angle'),
]

#loading = PointLoading(blade, wind_table, None)
thrust_time  = [0, 1, 2,    10  ]
thrust_force = [0, 0, 10e3, 10e3]
thrust = np.c_[ thrust_force, np.zeros((4,2)) ].T
loadfunc = scipy.interpolate.interp1d(thrust_time, thrust)

# Modal element
base = RigidConnection('base', rotation=rotmat_y(-np.pi/2))
el = ModalElement('el', modes, distal=True, damping_freqs=combined_freqs)
rna = RigidBody('rna', 109993, np.diag([19406630, 19406630, 38813240]), nodal_load=loadfunc)
base.add_leaf(el)
el.add_leaf(rna)
system = System(base)

integ = Integrator(system)
integ.add_output(el.output_deflections())
integ.add_output(el.output_rotations())
integ.add_output(dynamics.LoadOutput(rna.iprox))

linsys = LinearisedSystem(system)

def ani_xy(s,t,y):
    return dynvis.anim(s, t, y, (0,1), (-5,45), (-5,5), velocities=False)

def ani_xz(s,t,y):
    return dynvis.anim(s, t, y, (0,2), (-5,45), (-5,5), velocities=False)

def ani_yz(s,t,y):
    return dynvis.anim(s, t, y, (1,2), (-10,10), (-10,20), velocities=False, only_free=True)


def doplot():
    fig = plt.figure()
    gs = gridspec.GridSpec(4, 2)
    
    fig.suptitle('Tower motion for point load at top')

    ax = fig.add_subplot(gs[0,0])
    ax.plot(thrust_time, thrust[0]/1000, 'b')
    ax.set_ylabel('Thrust (kN)')
    
    ax = fig.add_subplot(gs[1:3,0])
    ax.plot(t, -100*y[1][:,2], 'b', label='Mine')
    ax.plot(t,  100*bladed_defl[:,0], 'k--', label='Bladed')
    ax.set_ylabel('x defl (cm)')
    ax.legend(frameon=False)
    #ax.set_ylim((0,0.045))
    plt.setp(ax.get_legend().get_texts(), fontsize='small')

    ax = fig.add_subplot(gs[1:3,1])
    ax.plot(t, y[2][:,1], 'b', label='Mine (midspan)')
    ax.plot(t, bladed_defl[:,1], 'k--', label='Bladed')
    ax.set_ylabel('y rot (rad)')
    
    ax = fig.add_subplot(gs[3,0])
    ax.plot(t, y[0][:,1])
    ax.set_ylabel('Transverse mode')
    
    ax = fig.add_subplot(gs[3,1])
    ax.plot(t, y[0][:,[4,6]])
    ax.set_ylabel('Rotation & normal modes')

def blah():
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

