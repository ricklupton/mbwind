# -*- coding: utf-8 -*-
"""
Created on Tue 01 May 2012

Test response of blade root loads in a rotating rotor to prescribed harmonic
base motion of the structure.

@author: Rick Lupton
"""

from __future__ import division
import sys
if '..' not in sys.path: sys.path.insert(0,'..')

import numpy as np
from numpy import pi, array, dot

import dynamics

import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

from turbine import Turbine

# Turn off gravity
dynamics.OPT_GRAVITY = True
dynamics.OPT_GEOMETRIC_STIFFNESS = True

# Create model
bladed_file = r'C:\Users\Rick Lupton\Dropbox\phd\Bladed\Models\OC3-Hywind_SparBuoy_NREL5MW.prj'
tb = Turbine(bladed_file, rigid=False)
tb.set_base_motion(4, 0.80, 10*pi/180)
#t,y = tb.simulate(rotor_speed=2, t1=60, dt=0.05, init=True)

def p(parts=False):
    fig = plt.figure()
    fig.set_size_inches(15,10,forward=True)
    gs = gridspec.GridSpec(6, 2)
    motion = ['surge','sway','heave','roll','pitch','yaw'][tb.base_motion]
    fig.suptitle('Blade loads in response to base motion: {}'.format(motion))
        
    assert motion == 'pitch'
    az = [y[1][:,0], y[6][:,0], y[11][:,0]]
    pitch = [y[0][:,tb.base_motion], y[5][:,tb.base_motion], y[10][:,tb.base_motion]]
    mh = tb.modes.mass * tb.tower.hubheight
    h = tb.tower.hubheight
    mrg = tb.modes.I0[0]
    I = tb.modes.J0[0,0]
    g = dynamics.gravity * dynamics.OPT_GRAVITY
    Fx = {
        r'$\dot\theta^2$':   -mrg*az[1]**2,
        r'$\dot\phi^2$ (1)': -pitch[1]**2 * mh*np.cos(az[0]),
        r'$\dot\phi^2$ (2)': -pitch[1]**2 * mrg/2*(1+np.cos(2*az[0])),
        r'$g$':               tb.modes.mass*g*np.cos(az[0])*np.cos(pitch[0]),
    }
    Fy = {
        r'$\ddot\theta$':    -mrg * az[2],
        r'$\dot\phi^2$ (1)': -pitch[1]**2 * mh*np.sin(az[0]),
        r'$\dot\phi^2$ (2)': -pitch[1]**2 * mrg/2*np.sin(2*az[0]),
        r'$g$':               tb.modes.mass*g*np.sin(az[0])*np.cos(pitch[0]),
    }
    Fz = {
        r'$\ddot\phi$':          -pitch[2]*(mh + mrg*np.cos(az[0])),
        r'$\dot\phi\dot\theta$':  2*mrg*np.sin(az[0])*az[1]*pitch[1],
        r'$g$':                   tb.modes.mass*g*np.sin(pitch[0]),
    }
    Mx = {' ': t*0}
    My = {
        r'$\ddot\phi$':           pitch[2]*(mrg*h + I*np.cos(az[0])),
        r'$\dot\phi\dot\theta$': -2*I*np.sin(az[0])*az[1]*pitch[1],
        r'$g$':                  -mrg*g*np.sin(pitch[0]),
    }
    Mz = {
        r'$\dot\theta^2$':   -I*az[2],
        r'$\dot\phi^2$ (1)': -pitch[1]**2 * mrg*h*np.sin(az[0]),
        r'$\dot\phi^2$ (2)': -pitch[1]**2 * I/2*np.sin(2*az[0]),
        r'$g$':               mrg*g*np.sin(az[0])*np.cos(pitch[0]),
    }
    
    ax = fig.add_subplot(gs[0,0])
    ax.plot(t, y[0][:,tb.base_motion], 'k')
    ax.set_ylabel('Base {} motion'.format(motion))
    #ax1.legend(frameon=False, loc='upper left')
    #plt.setp(ax1.get_legend().get_texts(), fontsize='small')

    ax = fig.add_subplot(gs[0,1])
    ax.plot(t, np.degrees(az[0]) % 360, 'k')
    ax.set_ylabel('Rotor azimuth')

    ax = fig.add_subplot(gs[1,0])
    ax.plot(t, y[21][:,1], 'g', alpha=1.0)
    ax.plot(t, y[22][:,1], 'g', alpha=0.6)
    ax.plot(t, y[23][:,1], 'g', alpha=0.3)
    ax.set_ylabel('IP deflection (m)')
    
    ax = fig.add_subplot(gs[1,1])
    ax.plot(t, y[21][:,2], 'b', alpha=1.0)
    ax.plot(t, y[22][:,2], 'b', alpha=0.6)
    ax.plot(t, y[23][:,2], 'b', alpha=0.3)
    ax.set_ylabel('OOP deflection (m)')
    
    loads = ['Axial force (kN)','IP force (kN)','OOP force (kN)',
             'Twist moment (kNm)','Edge moment (kNm)','Flap moment (kNm)']
    colours = ['r','g','b','r','g','b']
    results = [0,1,2,3,5,4]
    expected = [Fx,Fy,Fz,Mx,Mz,My] # NB order
    for i in range(6):
        ax = fig.add_subplot(gs[2+i%3,i//3])
        if parts:
            ax.plot(t, y[18][:,results[i]]/1e3, colours[i])
            for style,(k,v) in zip(('k--','m--','c--','y--'),expected[i].items()):
                ax.plot(t, v/1e3, style, label=k)
            ax.legend(frameon=True, loc='upper right')
            plt.setp(ax.get_legend().get_texts(), fontsize='small')
        else:
            expec = np.sum(expected[i].values(), axis=0)
            ax.plot(t, y[18][:,results[i]]/1e3, colours[i], t, expec/1e3, 'k--')
        ax.set_ylabel(loads[i])
    
    ax = fig.add_subplot(gs[5,0])
    expec = np.sqrt(np.sum(expected[1].values(), axis=0)**2 + \
                    np.sum(expected[2].values(), axis=0)**2   )
    res = np.sqrt(y[18][:,results[1]]**2 + y[18][:,results[2]]**2)
    ax.plot(t, res/1e3, 'b', t, expec/1e3, 'k--')
    
    ax = fig.add_subplot(gs[5,1])
    expec = np.sqrt(np.sum(expected[4].values(), axis=0)**2 + \
                    np.sum(expected[5].values(), axis=0)**2   )
    res = np.sqrt(y[18][:,results[4]]**2 + y[18][:,results[5]]**2)
    ax.plot(t, res/1e3, 'b', t, expec/1e3, 'k--')
        
    ax.set_xlabel('Time / s')
    
