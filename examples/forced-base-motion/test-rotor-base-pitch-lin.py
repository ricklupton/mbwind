# -*- coding: utf-8 -*-
"""
Created on Tue 01 May 2012

Test response of blade root loads in a rotating rotor to a harmonic force at
the base of the structure.

@author: Rick Lupton
"""

from __future__ import division

import numpy as np
from numpy import pi, array, dot
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

import dynamics
from turbine import Turbine

# Turn off gravity
dynamics.OPT_GRAVITY = True
dynamics.OPT_GEOMETRIC_STIFFNESS = True

# Parameters
pitch_freq  = 0.3 # rad/s
pitch_amp   = 0.3 # rad
rotor_speed = 2   # rad/s

# Create model
bladed_file = r'C:\Users\Rick Lupton\Dropbox\phd\Bladed\Models\OC3-Hywind_SparBuoy_NREL5MW.prj'
tb = Turbine(bladed_file)

# Linearise model and simulate
lin = tb.lin(az0=0, rotor_speed=rotor_speed, init=True)
t,yl = lin.integrate(90)

# Convert to MBC and simulate
mbclin = lin.multiblade_transform((0, rotor_speed), [range(0,4),range(4,8),range(8,12)])
t,ym = mbclin.integrate(90)

# Simulate original full system
#t,y = tb.simulate(rotor_speed=2, t1=90, dt=0.10, init=True)


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
    
