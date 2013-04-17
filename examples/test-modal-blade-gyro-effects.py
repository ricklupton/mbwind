# -*- coding: utf-8 -*-
"""
Created on Fri 13 Apr 2012

Test modal element gyroscopic loading effects:
    rotate blade and constant speed then apply a yaw rate

@author: Rick Lupton
"""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import dynamics
from dynamics import (System, ModalElement, Hinge,
                      RigidConnection, rotmat_x, rotmat_y, Integrator)
from linearisation import ModalRepresentation
import dynvis

dynamics.gravity = 0

bladedpath = '/bladed/gyro_effects/simpleblade_10rpm_10degs'

# Modal element using data from Bladed model
print "Loading modes from '%s'..." % bladedpath
modes = ModalRepresentation.from_Bladed(bladedpath+'.$pj')

rotorspeed = 10 * np.pi/30 # 10 rpm
overhang = 1 # 1m

# Modal element
yaw = Hinge('yaw', [0,0,1])
overhang = RigidConnection('overhang', [-overhang,0,0])
bearing = Hinge('bearing', [1,0,0],np.dot(rotmat_y(-np.pi/2), rotmat_x(-np.pi/2)))
el = ModalElement('el', modes)
yaw.add_leaf(overhang)
overhang.add_leaf(bearing)
bearing.add_leaf(el)
system = System(yaw)

# Prescribe rotation speed
system.prescribe(bearing.istrain, 0.0)
system.qd[bearing.istrain] = rotorspeed

# Prescribed yaw angle
acc_period = 0.1
yaw_time1 = 3.0
yaw_time2 = 6.0
dvyaw = 10 * np.pi/180
system.prescribe(yaw.istrain,
    acc=lambda t: (t>=yaw_time1 and t<(yaw_time1+acc_period)) and  dvyaw/acc_period or 0 +\
                  (t>yaw_time2 and t<=(yaw_time2+acc_period)) and -dvyaw/acc_period or 0)
system.find_equilibrium()

integ = Integrator(system, ())
integ.add_position_output(yaw.istrain, 'yaw position')
integ.add_velocity_output(yaw.istrain, 'yaw velocity')
integ.add_acceleration_output(yaw.istrain, 'yaw acceleration')
integ.add_acceleration_output(el.imult)
integ.add_force_output(bearing.idist, "root load")
integ.add_custom_output(lambda s: el.station_positions()[-1], 'tip pos')
integ.add_custom_output(lambda s: el.modes.X(el.xstrain)[-1,1:3], 'tip defl')
integ.add_custom_output(lambda s: el.quad_forces, 'quad_forces')
integ.add_custom_output(lambda s: el.quad_stress, 'quad_stress')

# Load Bladed data for comparison
import pybladed.data
brun = pybladed.data.BladedRun(bladedpath)
bladed_defl = np.c_[
    brun.get('blade 1 x-deflection;4'),
    brun.get('blade 1 y-deflection;4'),
    brun.get('blade 1 x-position;4'),
    brun.get('blade 1 y-position;4'),
    brun.get('blade 1 z-position;4'),
    brun.get('rotor azimuth'),
    brun.get('nacelle yaw displacement'),
    brun.get('blade 1 root Fz'),
    brun.get('blade 1 root Fx'),
    brun.get('blade 1 root Fy'),
    brun.get('blade 1 root Mz'),
    brun.get('blade 1 root Mx'),
    brun.get('blade 1 root My'),
]

def ani_xy(s,t,y):
    return dynvis.anim(s, t, y, (0,1), (-5,45), (-5,5), velocities=False)

def ani_xz(s,t,y):
    return dynvis.anim(s, t, y, (0,2), (-5,45), (-5,5), velocities=False)

def ani_yz(s,t,y):
    return dynvis.anim(s, t, y, (1,2), (-10,10), (-10,20), velocities=False, only_free=True)

def doplot():
    b = bladed_defl[:len(t),:]
    
    fig = plt.figure()
    fig.set_size_inches(15,10,forward=True)
    gs = gridspec.GridSpec(4, 2)
    
    fig.suptitle('Gyroscopic yaw effects')
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(t, y[:,9], 'k', label='Mine', alpha=0.5)
    ax1.plot(t, b[:,2], 'k--', label='Bladed')
    ax1.set_ylabel('x-position / m')
    ax1.legend(frameon=False, loc='upper left')
    plt.setp(ax1.get_legend().get_texts(), fontsize='small')

    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(t, y[:,10], 'k', t, b[:,3], 'k--', alpha=0.5)
    ax2.set_ylabel('y-position / m')
    
    ax3 = fig.add_subplot(gs[2,0])
    ax3.plot(t, y[:,11], 'k', t, b[:,4]-61.5, 'k--', alpha=0.5)
    ax3.set_ylabel('z-position / m')
    
    ax4 = fig.add_subplot(gs[3,0])
    ax4.plot(t, y[:,0], 'k', t, b[:,6], 'k--', alpha=0.5)
    ax4.set_xlabel('Time / s')
    ax4.set_ylabel('Yaw angle')
    
    ax5 = fig.add_subplot(gs[0,1])
    ax5.plot(t, y[:,12], 'k', t, b[:,0], 'k--', alpha=0.5)
    ax5.set_ylabel('OOP deflection / m')
    
    ax6 = fig.add_subplot(gs[1,1])
    ax6.plot(t, y[:,13], 'k', t, b[:,1], 'k--', alpha=0.5)
    ax6.set_ylabel('IP deflection / m')
    
    ax7 = fig.add_subplot(gs[2,1])
    ax7.set_color_cycle(['r','g','b'])
    ax7.plot(t, y[:,3:6]/1e3, t, b[:,7:10]/1e3, '--', alpha=0.5)
    ax7.set_ylabel('Root forces / kN')
    ax7.legend(('x','y','z'), frameon=False, loc='upper left')
    plt.setp(ax7.get_legend().get_texts(), fontsize='small')
    
    ax7 = fig.add_subplot(gs[3,1])
    ax7.set_color_cycle(['r','g','b'])
    ax7.plot(t, y[:,6:9]/1e3, t, b[:,10:13]/1e3, '--', alpha=0.5)
    ax7.set_ylabel('Root moments / kNm')
    
if True:
    # Run a simluation
    t,y = integ.integrate(10, 0.05)


assert np.allclose(y[:,0], bladed_defl[:,6], 1e-3, 1e-2), "yaw angle"
#assert np.allclose(y[:,3], bladed_defl[:,2], 1e-2, 1e-2), "x-pos"
#assert np.allclose(y[:,4], bladed_defl[:,3]), "y-pos"
#assert np.allclose(y[:,5], bladed_defl[:,4]-61.5), "z-pos"
#assert np.allclose(y[:,6], bladed_defl[:,0]), "x-defl"
#assert np.allclose(y[:,7], bladed_defl[:,1]), "y-defl"
