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
from dynamics import (System, Hinge, RigidConnection, FreeJoint, RigidBody,
                      ModalElement, Integrator, rotmat_x, rotmat_y)
import linearisation
import dynvis
from blade import Blade, Tower

from pandas import Index, DataFrame, Series

class TurbineResults(object):
    def __init__(self, t, y, blade_mode_names):
                 
        base = y[0]
        azimuth = y[1]
        blade_strains = y[2:5]
        shub = y[17]
        rhub = y[18]
        root_loads = y[19:22]
        tip_defls = y[22:25]        
        all_strains = np.c_[ tuple(y[:5]) ]
                     
        self.t = t
        idx = Index(t, name='Time')
        
        self.strains = DataFrame(all_strains, index=idx)
        
        self.base = DataFrame(base, index=idx, columns=['surge', 'sway', 'heave',
                                                      'roll', 'pitch', 'yaw'])
        
        self.azimuth = Series(azimuth[:,0], index=idx)
        
        # Hub loads, rotating and stationary, GL coords
        self.shub = DataFrame(shub, columns=['Fx','Fy','Fz','Mx','My','Mz'], index=idx)
        self.rhub = DataFrame(rhub, columns=['Fx','Fy','Fz','Mx','My','Mz'], index=idx)
        
        # Blade root loads, GL coords
        self.root_loads = []
        for rl in root_loads:
            self.root_loads.append(DataFrame({
                'Fx': -rl[:,2],
                'Fy':  rl[:,1],
                'Fz':  rl[:,0],
                'Mx': -rl[:,5],
                'My':  rl[:,4],
                'Mz':  rl[:,3],
            }, index=idx))
        self.root = self.root_loads[0]
        
        # Tip deflections in GL coords
        self.tip_defls = []
        for td in tip_defls:
            self.tip_defls.append(DataFrame({
                'x': -td[:,2],
                'y':  td[:,1],
                'z':  td[:,0],
            }, index=idx))
        self.tip = self.tip_defls[0]
        
        # Blade modal amplitudes
        self.blade_strains = []
        for bs in blade_strains:
            self.blade_strains.append(DataFrame(bs, index=idx, columns=blade_mode_names))
        

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
        self.integ.add_output(dynamics.LoadOutput(self.bearing.iprox, local=True))
        self.integ.add_output(dynamics.LoadOutput(self.bearing.idist[0], local=True))
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
        t,y = self.integ.integrate(t1, dt, t0)
        #for i,lab in enumerate(self.integ.labels):
        #    print "%2d  %s" % (i,lab)
        #return self.t, self.y
        
        # Build results structure
        results = TurbineResults(t, y, self.modes.mode_names)
        return results
    
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

    def ani(self, results, x=0, y=1):
        H = self.tower.hubheight + self.blade.radii[-1]
        limits = [(-H,H), (-H,H), (-5,H+5)]
        return dynvis.anim(self.system, results.t, results.strains.as_matrix(),
                           (x, y), limits[x], limits[y])
                           
    #def ani(self, vs=(0,1), t=None, y=None):
    #    if t is None: t = self.t
    #    if y is None: y = self.y
    #    limits = [(-10,10), (-42,42), (-5,110)]
    #    return dynvis.anim(self.system, t, y, vs, limits[vs[0]], limits[vs[1]])

class FlappedTurbineResults(object):
    def __init__(self, t, y):
                 
        base = y[0]
        azimuth = y[1]
        flap_strains = y[2:5]
        shub = y[17]
        rhub = y[18]
        root_loads = y[19:22]
        all_strains = np.c_[ tuple(y[:5]) ]
                     
        self.t = t
        idx = Index(t, name='Time')
        
        self.strains = DataFrame(all_strains, index=idx)
        
        self.base = DataFrame(base, index=idx, columns=['surge', 'sway', 'heave',
                                                      'roll', 'pitch', 'yaw'])
        
        self.azimuth = Series(azimuth[:,0], index=idx)
        
        # Hub loads, rotating and stationary, GL coords
        self.shub = DataFrame(shub, columns=['Fx','Fy','Fz','Mx','My','Mz'], index=idx)
        self.rhub = DataFrame(rhub, columns=['Fx','Fy','Fz','Mx','My','Mz'], index=idx)
        
        # Blade root loads, GL coords
        self.root_loads = []
        for rl in root_loads:
            self.root_loads.append(DataFrame({
                'Fx': -rl[:,2],
                'Fy':  rl[:,1],
                'Fz':  rl[:,0],
                'Mx': -rl[:,5],
                'My':  rl[:,4],
                'Mz':  rl[:,3],
            }, index=idx))
        self.root = self.root_loads[0]
        
        # Blade modal amplitudes
        self.flap_strains = []
        for bs in flap_strains:
            self.flap_strains.append(DataFrame(bs, index=idx, columns=['Flap angle']))
        

class FlappedBladeTurbine(object):
    def __init__(self, bladed_file, root_length=0):
        # Load modes but use a simple flapped blade
        print "Loading modes from '%s'..." % bladed_file
        self.blade = Blade(bladed_file)
        self.tower = Tower(bladed_file)
        self.modes = self.blade.modal_rep()

        # Calculate equivalent blade properties
        I1 = self.modes.I0[0]
        I2 = self.modes.J0[0,0]
        print I1, I2
        wflap = self.modes.freqs[0]
        bmass = self.modes.mass
        inertia = self.modes.inertia_tensor(np.zeros(len(self.modes.freqs)))
        Xc = [I1 / bmass, 0, 0]
        kflap = I2 * wflap**2

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
        self.flap1 = Hinge('flap1', [0,1,0])
        self.flap2 = Hinge('flap2', [0,1,0])
        self.flap3 = Hinge('flap3', [0,1,0])
        self.blade1 = RigidBody('blade1', bmass, inertia, Xc)
        self.blade2 = RigidBody('blade2', bmass, inertia, Xc)
        self.blade3 = RigidBody('blade3', bmass, inertia, Xc)
        
        self.flap1.stiffness = self.flap2.stiffness = self.flap3.stiffness = kflap
        
        self.base.add_leaf(self.towerlink)
        self.towerlink.add_leaf(self.bearing)
        self.bearing.add_leaf(root1)
        self.bearing.add_leaf(root2)
        self.bearing.add_leaf(root3)
        root1.add_leaf(self.flap1)
        root2.add_leaf(self.flap2)
        root3.add_leaf(self.flap3)
        self.flap1.add_leaf(self.blade1)
        self.flap2.add_leaf(self.blade2)
        self.flap3.add_leaf(self.blade3)
        self.system = System(self.base)

        # Prescribed DOF accelerations - constant rotor speed
        self.base_motion = None
        self.base_motion_amp = 0
        self.system.prescribe(self.bearing, vel=0)
        self.system.prescribe(self.base, vel=0)
        
        # setup integrator
        self.integ = Integrator(self.system, ('pos','vel','acc'))
        self.integ.add_output(dynamics.LoadOutput(self.base.iprox))
        self.integ.add_output(dynamics.LoadOutput(self.towerlink.iprox))
        self.integ.add_output(dynamics.LoadOutput(self.bearing.iprox, local=True))
        self.integ.add_output(dynamics.LoadOutput(self.bearing.idist[0], local=True))
        for b in (self.blade1, self.blade2, self.blade3):
            self.integ.add_output(dynamics.LoadOutput(b.iprox, local=True))
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
        if qm0 is not None:  # initial hinge angle
            self.system.q[self.flap1.istrain][0] = qm0
            self.system.q[self.flap2.istrain][0] = qm0
            self.system.q[self.flap3.istrain][0] = qm0
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
        t,y = self.integ.integrate(t1, dt, t0)
        #for i,lab in enumerate(self.integ.labels):
        #    print "%2d  %s" % (i,lab)
        #return self.t, self.y
        
        # Build results structure
        results = FlappedTurbineResults(t, y)
        return results
    
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

    def ani(self, results, x=0, y=1):
        H = self.tower.hubheight + self.blade.radii[-1]
        limits = [(-H,H), (-H,H), (-5,H+5)]
        return dynvis.anim(self.system, results.t, results.strains.as_matrix(),
                           (x, y), limits[x], limits[y])
                           