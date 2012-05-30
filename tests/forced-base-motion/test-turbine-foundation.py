# -*- coding: utf-8 -*-
"""
Created on Fri 11 May 2012

Test response of rigid turbine structure to applied forces on an elastic
foundation, defined by a stiffness matrix.

@author: Rick Lupton
"""

from __future__ import division

import numpy as np
from numpy import pi, array, dot
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

import dynamics
from turbine import Turbine

# Options
dynamics.OPT_GRAVITY = True
dynamics.OPT_GEOMETRIC_STIFFNESS = False

# Create model
bladed_file = r'C:\Users\Rick Lupton\Dropbox\phd\Bladed\Models\OC3-Hywind_SparBuoy_NREL5MW.prj'
tb = Turbine(bladed_file, rigid=False)

# Set base motion
tb.system.free(tb.base)
tb.system.prescribe(tb.base, acc=0, vel=0, part=[0,1,2,3,5])

# Define foundation matrix - no coupling for now
# Choose to match appropriate frequencies of rigid-body motion (Karimirad & Moan, 2012)
#  in surge, sway, heave, roll, pitch yaw respectively
rigid_body_freqs = array([0.05, 0.05, 0.20, 0.22, 0.22, 0.84]) **2
foundation = np.diag([
    tb.mass * rigid_body_freqs[0],
    tb.mass * rigid_body_freqs[1],
    tb.mass * rigid_body_freqs[2],
    tb.inertia[0,0] * rigid_body_freqs[3],
    tb.inertia[1,1] * rigid_body_freqs[4],
    tb.inertia[2,2] * rigid_body_freqs[5] + 100
])
tb.base.stiffness = foundation*1000

# Parameters
rotor_speed = 2
t = np.arange(0, 90, 0.25)

# Linearise model and make MBC version
#lin = tb.lin(az0=0, rotor_speed=rotor_speed, init=False)
#mbclin = lin.multiblade_transform((0, rotor_speed), [range(0,4),range(4,8),range(8,12)])

# Set initial pitch deflection and simulate
z0 = np.zeros(13)
z0[0] = 0.3
#yl = lin.integrate(t, z0)
#ym = mbclin.integrate(t, z0)

# Simulate original full system
tb.base_motion = 4
tb.base_motion_amp = 0.3
tt,y = tb.simulate(t1=90, dt=0.25, rotor_speed=rotor_speed, init=False)


def mbc(az, u):
    N = len(az)
    B = array([
        np.ones((3,N)),
        [2*np.cos(az), 2*np.cos(az+2*pi/3), 2*np.cos(az+4*pi/3)],
        [2*np.sin(az), 2*np.sin(az+2*pi/3), 2*np.sin(az+4*pi/3)],
    ]) / 3
    u = array(u)
    z = np.einsum('zbt,btj->jzt', B, u)
    return z

def p(parts=False):
    fig = plt.figure()
    fig.set_size_inches(15,10,forward=True)
    gs = gridspec.GridSpec(4, 1)
       
    az = y[1][:,0]
    z = mbc(az, y[2:5])
      
    ax = None
    for imode in range(4):
        ax = fig.add_subplot(gs[imode,0], sharex=ax)
        ax.plot(t,y[2][:,imode],'k',alpha=0.8)
        ax.plot(t,y[3][:,imode],'k',alpha=0.4)
        ax.plot(t,y[4][:,imode],'k',alpha=0.2)
        ax.plot(t,z[imode].T)
        for i in np.nonzero(np.diff(az) < 0)[0]:
            ax.axvline(t[i], alpha=0.1)
        ax.axhline(0, alpha=0.1)            
        ax.set_title(tb.modes.mode_descriptions[imode])
        #ax1.legend(frameon=False, loc='upper left')
        #plt.setp(ax1.get_legend().get_texts(), fontsize='small')
            
    ax.set_xlabel('Time / s')
    
