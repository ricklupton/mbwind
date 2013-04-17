# -*- coding: utf-8 -*-
"""
Created on Thu 26 Apr 2012

Test non-linearity of a rotating modal element.

@author: Rick Lupton
"""

from __future__ import division
import sys
if '..' not in sys.path: sys.path.insert(0,'..')

import numpy as np
from numpy import pi, array, dot

import dynamics
from dynamics import (System, Hinge, DirectModalElement, RigidConnection,
                      ModalElement, Integrator, rotmat_x, rotmat_y)
import dynvis
import linearisation
from blade import Blade

import matplotlib.pylab as plt

class Rotor(object):
    def __init__(self, mode_source_file, root_length=0):
        # Modal element using data from Bladed model
        print "Loading modes from '%s'..." % mode_source_file
        self.blade = Blade(mode_source_file)
        self.modes = self.blade.modal_rep()

        Ry = rotmat_y(-pi/2)
        Rhb1 = rotmat_x(0 * 2*pi/3)
        Rhb2 = rotmat_x(1 * 2*pi/3)
        Rhb3 = rotmat_x(2 * 2*pi/3)

        self.bearing = Hinge('bearing', [1,0,0])
        root1 = RigidConnection('root1', root_length*np.dot(Rhb1,[0,0,1]), dot(Rhb1,Ry))
        root2 = RigidConnection('root2', root_length*np.dot(Rhb2,[0,0,1]), dot(Rhb2,Ry))
        root3 = RigidConnection('root3', root_length*np.dot(Rhb3,[0,0,1]), dot(Rhb3,Ry))
        self.blade1 = ModalElement('blade1', self.modes)
        self.blade2 = ModalElement('blade2', self.modes)
        self.blade3 = ModalElement('blade3', self.modes)
        
        self.bearing.add_leaf(root1)
        self.bearing.add_leaf(root2)
        self.bearing.add_leaf(root3)
        root1.add_leaf(self.blade1)
        root2.add_leaf(self.blade2)
        root3.add_leaf(self.blade3)
        self.system = System(self.bearing)

        # Prescribed DOF accelerations - constant rotor speed
        self.system.prescribe(self.bearing, acc=0.0)
        
        # setup integrator
        self.integ = Integrator(self.system, ('pos','vel'))
        self.integ.add_output(dynamics.LoadOutput(self.bearing.iprox))
        self.integ.add_output(dynamics.LoadOutput(self.blade1.iprox, local=True))
        self.integ.add_output(self.blade1.output_deflections())
        self.integ.add_output(self.blade1.output_positions())
        
    def simulate(self, qm0=None, spin=10.0, t1=1.5, dt=0.01):
        # reset
        self.system.q [:] = 0.0
        self.system.qd[:] = 0.0

        # initial conditions
        if qm0 is not None:  # initial modal amplitudes
            self.system.q[self.blade1.istrain] = qm0
            self.system.q[self.blade2.istrain] = qm0
            self.system.q[self.blade3.istrain] = qm0
        self.system.qd[self.bearing.istrain][0] = spin
        
        # simulate
        self.t,self.y = self.integ.integrate(t1, dt)
        for i,lab in enumerate(self.integ.labels()):
            print "%2d  %s" % (i,lab)
        return self.t, self.y
    
    def lin(self, qm0=None, spin=10.0):
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
        self.system.prescribe(self.bearing, vel=spin, acc=0.0)
        self.system.qd[self.bearing.istrain][0] = spin
        
        linsys = linearisation.LinearisedSystem(self.system, qm0)
        return linsys

    def ani(self, t=None, y=None, planview=True):
        if t is None: t = self.t
        if y is None: y = self.y
        l = 40
        if planview:
            return dynvis.anim(self.system, t, y, (1,2), (-l,l), (-l,l))
        else:
            return dynvis.anim(self.system, t, y, (0,1), (-l,l), (-5,5))

# Turn off gravity
dynamics.OPT_GRAVITY = False
dynamics.OPT_GEOMETRIC_STIFFNESS = True

# Create model
bladed_file = '/bladed/uniform_blade_2modes.prj'
rotor = Rotor(bladed_file, 0)
#t,y2 = rotor.simulate([0.2, 0.0], spin=10, t1=1, dt=0.01)

def measure_period(t,y):
    t_cross = t[np.nonzero(np.diff(y>=0))]
    print t_cross
    if len(t_cross) > 2: period = np.diff(t_cross[-3::2])[0]
    elif len(t_cross) == 2: period = 2*np.diff(t_cross)[0]
    else: period = np.nan
    return period

def test_speeds(speeds):
    print '----------------'
    print 'MODAL ROTOR TEST'
    print '----------------'
    print
    print 'Varying speeds:'
    print speeds
    print
    
    # results
    ip = []
    oop = []
    for Omega in speeds:
        print '::: Omega = {}'.format(Omega)
        T = 2*pi / max(Omega, rotor.modes.freqs[0])
        print T, T*4, T/30
                
        # out-of-plane
        print '{}: '.format(rotor.modes.mode_descriptions[0]),
        t,y = rotor.simulate([0.0, 0.02], spin=Omega, t1=T*4, dt=T/30)
        ip.append(2*np.pi/measure_period(t,y[1][:,1]))

        # out-of-plane
        print '{}: '.format(rotor.modes.mode_descriptions[1]),
        t,y = rotor.simulate([0.02, 0.0], spin=Omega, t1=T*4, dt=T/30)
        oop.append(2*np.pi/measure_period(t,y[1][:,0]))
        print

    return np.array([ip,oop]), t,y

def plotfreqs(speeds, freqs):
    assert speeds[0] == 0
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(speeds, freqs[0], '.b-', label='IP')
    ax.plot(speeds, freqs[1], '.g-', label='OOP')
    ax.plot(speeds, np.ones_like(speeds)*freqs[0,0], 'k--')
    ax.plot(speeds, np.ones_like(speeds)*freqs[1,0], 'k--')
    ax.plot(speeds, speeds, 'k--')
    ax.set_title('Frequency of in-plane and out-of-plane oscillations')
    ax.set_xlabel('Rotational speed (rad/s)')
    ax.set_ylabel('Vibration frequency (rad/s)')
    ax.legend(loc='upper left')

##### Test frequencies from time-domain results, with two root lengths ######
if False:
    speeds = [0, 1, 5, 10, 15, 20, 30, 40, 50]
    rotor0 = Rotor(bladed_file, 0)
    freqs0,t,y = test_speeds(speeds)
    #rotor10 = Rotor(bladed_file, 10)
    #freqs10,t,y = test_speeds(speeds)
    
    # At the moment, expect root length not to affect result because modes do
    # not include any radial deflection
    
    plotfreqs(speeds, freqs0)
    #plotfreqs(speeds, freqs10)

##### Test linearisation ######
def campbell_diagram(speeds):
    speeds = np.array(speeds)
    print '----------------------------'
    print 'Calculating Campbell diagram'
    print '----------------------------'
    print
    print 'Varying speeds:'
    print speeds
    print
    
    results = []
    for Omega in speeds:
        print '::: Omega = {}'.format(Omega)
        linsys = rotor.lin(spin=Omega)
        w,v = linsys.modes()
        results.append(w)
    results = np.array(results)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(speeds, results)
    ax.plot(speeds, speeds, 'k--', speeds, 2*speeds, 'k--')
    ax.set_title('Campbell diagram')
    ax.set_xlabel('Rotational speed (rad/s)')
    ax.set_ylabel('Vibration frequency (rad/s)')
    ax.legend(rotor.modes.mode_descriptions, loc='upper left')
    return results