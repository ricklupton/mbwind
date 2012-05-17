# -*- coding: utf-8 -*-
"""
Created on Wed May 09 15:40:08 2012

Simple floating turbine model

@author: Rick Lupton
"""

from __future__ import division

import numpy as np
from numpy import pi, dot

import dynamics
from dynamics import (System, Hinge, RigidConnection, FreeJoint,
                      ModalElement, Integrator, rotmat_x, rotmat_y)
import linearisation
import dynvis
from blade import Blade, Tower


class Turbine(object):
    def __init__(self, bladed_file, root_length=0, rigid=False):
        # Modal element using data from Bladed model
        print "Loading modes from '%s'..." % bladed_file
        self.blade = Blade(bladed_file)
        self.tower = Tower(bladed_file)
        self.modes = self.blade.modal_rep()

        Ry = rotmat_y(-pi/2)
        Rhb1 = rotmat_x(0 * 2*pi/3)
        Rhb2 = rotmat_x(1 * 2*pi/3)
        Rhb3 = rotmat_x(2 * 2*pi/3)

        self.base = FreeJoint('base')
        self.towerlink = RigidConnection('tower', [0,0,self.tower.hubheight])
        self.bearing = Hinge('bearing', [1,0,0])
        root1 = RigidConnection('root1', root_length*np.dot(Rhb1,[0,0,1]), dot(Rhb1,Ry))
        root2 = RigidConnection('root2', root_length*np.dot(Rhb2,[0,0,1]), dot(Rhb2,Ry))
        root3 = RigidConnection('root3', root_length*np.dot(Rhb3,[0,0,1]), dot(Rhb3,Ry))
        self.blade1 = ModalElement('blade1', self.modes)
        self.blade2 = ModalElement('blade2', self.modes)
        self.blade3 = ModalElement('blade3', self.modes)
        
        self.base.add_leaf(self.towerlink)
        self.towerlink.add_leaf(self.bearing)
        self.bearing.add_leaf(root1)
        self.bearing.add_leaf(root2)
        self.bearing.add_leaf(root3)
        root1.add_leaf(self.blade1)
        root2.add_leaf(self.blade2)
        root3.add_leaf(self.blade3)
        self.system = System(self.base)

        # Prescribed DOF accelerations - constant rotor speed
        self.base_motion = None
        self.base_motion_amp = 0
        self.system.prescribe(self.bearing, vel=0)
        self.system.prescribe(self.base, vel=0)
        if rigid:
            for b in (self.blade1, self.blade2, self.blade3):
                self.system.prescribe(b, vel=0)
        
        # setup integrator
        self.integ = Integrator(self.system, ('pos','vel','acc'))
        self.integ.add_output(dynamics.LoadOutput(self.base.iprox))
        self.integ.add_output(dynamics.LoadOutput(self.towerlink.iprox))
        self.integ.add_output(dynamics.LoadOutput(self.bearing.iprox))
        for b in (self.blade1, self.blade2, self.blade3):
            self.integ.add_output(dynamics.LoadOutput(b.iprox, local=True))
        for b in (self.blade1, self.blade2, self.blade3):
            self.integ.add_output(b.output_deflections())
        for b in (self.blade1, self.blade2, self.blade3):
            self.integ.add_output(b.output_positions())
        for b in (self.blade1, self.blade2, self.blade3):
            self.integ.add_output(dynamics.NodeOutput(b.iprox, local=True, deriv=2))

    def set_base_motion(self, dof, w, amp):
        self.base_motion = dof
        self.base_motion_amp = amp
        self.system.prescribe(self.base, part=dof,
                              vel=lambda t: -w   *amp*np.sin(w*t),
                              acc=lambda t: -w**2*amp*np.cos(w*t))

    def set_initial_conditions(self, qm0=None, az0=None, rotor_speed=None):
        # initial conditions
        if qm0 is not None:  # initial modal amplitudes
            self.system.q[self.blade1.istrain] = qm0
            self.system.q[self.blade2.istrain] = qm0
            self.system.q[self.blade3.istrain] = qm0
        if az0 is not None:
            self.system.q[self.bearing.istrain][0] = az0
        if self.base_motion is not None:
            self.system.q[self.base.istrain][self.base_motion] = self.base_motion_amp
        if rotor_speed is not None:
            self.system.prescribe(self.bearing, vel=rotor_speed)

    def simulate(self, qm0=None, az0=0.0, rotor_speed=10.0, t1=None,
                 dt=0.01, t0=0.0, init=False):
        # reset
        self.system.q [:] = 0.0
        self.system.qd[:] = 0.0
        self.set_initial_conditions(qm0, az0, rotor_speed)
        
        if t1 is None:
            t1 = 4*pi/rotor_speed if (rotor_speed != 0.0) else 2
        
        if init:
            self.system.find_equilibrium()
        
        # simulate
        self.t,self.y = self.integ.integrate(t1, dt, t0)
        for i,lab in enumerate(self.integ.labels()):
            print "%2d  %s" % (i,lab)
        return self.t, self.y
    
    def lin(self, qm0=None, az0=None, rotor_speed=None, init=False):
        # reset
        self.system.q [:] = 0.0
        self.system.qd[:] = 0.0
        self.set_initial_conditions(qm0, az0, rotor_speed)
        
        if init:
            self.system.find_equilibrium()
            
        # need initial amplitudes for linearisation point
        if qm0 is None: 
            qm0 = self.system.q[self.blade1.istrain]
        linsys = linearisation.LinearisedSystem(self.system, np.tile(qm0, 3))
        return linsys

    def ani(self, vs=(0,1), t=None, y=None):
        if t is None: t = self.t
        if y is None: y = self.y
        limits = [(-10,10), (-42,42), (-5,110)]
        return dynvis.anim(self.system, t, y, vs, limits[vs[0]], limits[vs[1]])
