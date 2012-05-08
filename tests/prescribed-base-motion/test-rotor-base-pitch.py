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
from dynamics import (System, Hinge, RigidConnection, FreeJoint,
                      ModalElement, Integrator, rotmat_x, rotmat_y)
import dynvis
import linearisation
from linearisation import ModalRepresentation

import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

class Turbine(object):
    def __init__(self, height, mode_source_file, root_length=0, rigid=False):
        # Modal element using data from Bladed model
        print "Loading modes from '%s'..." % mode_source_file
        self.modes = ModalRepresentation.from_Bladed(mode_source_file)
        self.tower_height = height

        Ry = rotmat_y(-pi/2)
        Rhb1 = rotmat_x(0 * 2*pi/3)
        Rhb2 = rotmat_x(1 * 2*pi/3)
        Rhb3 = rotmat_x(2 * 2*pi/3)

        self.base = FreeJoint('base')
        self.tower = RigidConnection('tower', [0,0,height])
        self.bearing = Hinge('bearing', [1,0,0])
        root1 = RigidConnection('root1', root_length*np.dot(Rhb1,[0,0,1]), dot(Rhb1,Ry))
        root2 = RigidConnection('root2', root_length*np.dot(Rhb2,[0,0,1]), dot(Rhb2,Ry))
        root3 = RigidConnection('root3', root_length*np.dot(Rhb3,[0,0,1]), dot(Rhb3,Ry))
        self.blade1 = ModalElement('blade1', self.modes)
        self.blade2 = ModalElement('blade2', self.modes)
        self.blade3 = ModalElement('blade3', self.modes)
        
        self.base.add_leaf(self.tower)
        self.tower.add_leaf(self.bearing)
        self.bearing.add_leaf(root1)
        self.bearing.add_leaf(root2)
        self.bearing.add_leaf(root3)
        root1.add_leaf(self.blade1)
        root2.add_leaf(self.blade2)
        root3.add_leaf(self.blade3)
        self.system = System(self.base)

        # Prescribed DOF accelerations - constant rotor speed
        self.system.prescribe(self.bearing, vel=0)
        self.system.prescribe(self.base, vel=0)
        if rigid:
            for b in (self.blade1, self.blade2, self.blade3):
                self.system.prescribe(b, vel=0)
        
        # setup integrator
        self.integ = Integrator(self.system, ('pos','vel','acc'))
        self.integ.add_output(dynamics.LoadOutput(self.base.iprox))
        self.integ.add_output(dynamics.LoadOutput(self.tower.iprox))
        self.integ.add_output(dynamics.LoadOutput(self.bearing.iprox))
        for b in (self.blade1, self.blade2, self.blade3):
            self.integ.add_output(dynamics.LoadOutput(b.iprox, local=True))
        for b in (self.blade1, self.blade2, self.blade3):
            self.integ.add_output(b.output_deflection())
        for b in (self.blade1, self.blade2, self.blade3):
            self.integ.add_output(b.output_positions())
        for b in (self.blade1, self.blade2, self.blade3):
            self.integ.add_output(dynamics.NodeOutput(b.iprox, local=True, deriv=2))

    def set_base_motion(self, dof, freq, amp):
        w = 2*pi*freq
        self.base_motion = dof
        self.base_motion_amp = amp
        self.system.prescribe(self.base, part=dof,
                              vel=lambda t: -w   *amp*np.sin(w*t),
                              acc=lambda t: -w**2*amp*np.cos(w*t))

    def simulate(self, qm0=None, az0=0.0, rotor_speed=10.0, t1=None,
                 dt=0.01, t0=0.0, init=False):
        # reset
        self.system.q [:] = 0.0
        self.system.qd[:] = 0.0

        # initial conditions
        if qm0 is not None:  # initial modal amplitudes
            self.system.q[self.blade1.istrain] = qm0
            self.system.q[self.blade2.istrain] = qm0
            self.system.q[self.blade3.istrain] = qm0
        self.system.q[self.bearing.istrain][0] = az0
        self.system.q[self.base.istrain][self.base_motion] = self.base_motion_amp
        #self.system.prescribe(self.bearing, vel=rotor_speed)
        self.system.qd[self.bearing.istrain][0] = rotor_speed
        
        if t1 is None:
            if rotor_speed != 0.0:
                t1 = 4*pi/rotor_speed
            else:
                t1 = 2
        
        if init:
            self.system.find_equilibrium()
        
        # simulate
        self.t,self.y = self.integ.integrate(t1, dt, t0)
        for i,lab in enumerate(self.integ.labels()):
            print "%2d  %s" % (i,lab)
        return self.t, self.y
    
    def lin(self, qm0=None, rotor_speed=10.0):
        # reset
        self.system.q [:] = 0.0
        self.system.qd[:] = 0.0

        # initial conditions
        if qm0 is not None:  # initial modal amplitudes
            self.system.q[self.blade1.istrain] = qm0
            self.system.q[self.blade2.istrain] = qm0
            self.system.q[self.blade3.istrain] = qm0
        else:
            qm0 = np.zeros(self.blade1._nstrain * 3)
        #self.system.prescribe(self.bearing, vel=rotor_speed)
        self.system.qd[self.bearing.istrain][0] = rotor_speed
        
        linsys = linearisation.LinearisedSystem(self.system, qm0)
        return linsys

    def ani(self, vs=(0,1), t=None, y=None):
        if t is None: t = self.t
        if y is None: y = self.y
        limits = [(-10,10), (-42,42), (-5,110)]
        return dynvis.anim(self.system, t, y, vs, limits[vs[0]], limits[vs[1]])

# Turn off gravity
dynamics.gravity = 0

# Create model
bladed_file = '/bladed/uniform_blade_2modes.prj'
tb = Turbine(60, bladed_file, 0, rigid=False)
tb.set_base_motion(4, 0.25, 0.3)
t,y = tb.simulate(rotor_speed=2, t1=60, dt=0.05, init=True)

def measure_period(t,y):
    t_cross = t[np.nonzero(np.diff(y>=0))]
    print t_cross
    if len(t_cross) > 2: period = np.diff(t_cross[-3::2])[0]
    elif len(t_cross) == 2: period = 2*np.diff(t_cross)[0]
    else: period = np.nan
    return period

def p(parts=False):
    fig = plt.figure()
    fig.set_size_inches(15,10,forward=True)
    gs = gridspec.GridSpec(5, 2)
    motion = ['surge','sway','heave','roll','pitch','yaw'][tb.base_motion]
    fig.suptitle('Blade loads in response to base motion: {}'.format(motion))
        
    assert motion == 'pitch'
    az = [y[1][:,0], y[6][:,0], y[11][:,0]]
    pitch = [y[0][:,tb.base_motion], y[5][:,tb.base_motion], y[10][:,tb.base_motion]]
    mh = tb.modes.mass * tb.tower_height
    h = tb.tower_height
    mrg = tb.modes.I0[0]
    I = tb.modes.J0[0,0]
    Fx = {
        r'$\dot\theta^2$':   -mrg*az[1]**2,
        r'$\dot\phi^2$ (1)': -pitch[1]**2 * mh*np.cos(az[0]),
        r'$\dot\phi^2$ (2)': -pitch[1]**2 * mrg/2*(1+np.cos(2*az[0])),
    }
    Fy = {
        r'$\ddot\theta$':    -mrg * az[2],
        r'$\dot\phi^2$ (1)': -pitch[1]**2 * mh*np.sin(az[0]),
        r'$\dot\phi^2$ (2)': -pitch[1]**2 * mrg/2*np.sin(2*az[0]),
    }
    Fz = {
        r'$\ddot\phi$':          -pitch[2]*(mh + mrg*np.cos(az[0])),
        r'$\dot\phi\dot\theta$':  2*mrg*np.sin(az[0])*az[1]*pitch[1],
    }
    Mx = {' ': t*0}
    My = {
        r'$\ddot\phi$':           pitch[2]*(mrg*h + I*np.cos(az[0])),
        r'$\dot\phi\dot\theta$': -2*I*np.sin(az[0])*az[1]*pitch[1],
    }
    Mz = {
        r'$\dot\theta^2$':   -I*az[2],
        r'$\dot\phi^2$ (1)': -pitch[1]**2 * mrg*h*np.sin(az[0]),
        r'$\dot\phi^2$ (2)': -pitch[1]**2 * I/2*np.sin(2*az[0]),
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
    ax.plot(t, y[21][:,1], 'g')
    ax.set_ylabel('IP deflection')
    
    ax = fig.add_subplot(gs[1,1])
    ax.plot(t, y[21][:,2], 'b')
    ax.set_ylabel('OOP deflection')
    
    loads = ['Axial force','IP force','OOP force','Twist moment','Edge moment','Flap moment']
    colours = ['r','g','b','r','g','b']
    results = [0,1,2,3,5,4]
    expected = [Fx,Fy,Fz,Mx,Mz,My] # NB order
    for i in range(6):
        ax = fig.add_subplot(gs[2+i%3,i//3])
        if parts:
            ax.plot(t, y[18][:,results[i]]/1e6, colours[i])
            for style,(k,v) in zip(('k:','k-.','k--','k-'),expected[i].items()):
                ax.plot(t, v/1e6, style, label=k)
            ax.legend(frameon=True, loc='upper right')
            plt.setp(ax.get_legend().get_texts(), fontsize='small')
        else:
            expec = np.sum(expected[i].values(), axis=0)
            ax.plot(t, y[18][:,results[i]]/1e6, colours[i], t, expec/1e6, 'k--')
        ax.set_ylabel(loads[i])
        
    ax.set_xlabel('Time / s')
    
