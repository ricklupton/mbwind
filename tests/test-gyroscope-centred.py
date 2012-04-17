# -*- coding: utf-8 -*-
"""
Created on Tue 3 Apr 2012

Test modelling of a gyroscope and precession/nutation, with the centre of 
gravity located at the centre of the gymbals, and a mass at the end

System consists of 3 hinges at origin about 3 axes to allow the gyroscope to
spin, tilt in elevation and rotate in azimuth.

The rigid spinning body is modelled in 3 ways:
    1. RigidBody element, mass and inertia specified directly.
    2. ModalElement element with no mode shapes, cross-section inertia and
       density distribution given.
    3. UniformBeam element with rigid strains, density and extra cross-section
       inertia given.

@author: Rick Lupton
"""

from __future__ import division
import sys
if '..' not in sys.path: sys.path.insert(0,'..')

import numpy as np

from dynamics import (System, Hinge, UniformBeam, RigidBody, RigidConnection,
                      ModalElement, solve_system, gravity, rotmat_y)
import dynvis
import linearisation

import matplotlib.pylab as plt

class Gyroscope(object):
    def __init__(self, length, radius, mass, spin, endmass):
        self.length = length
        self.radius = radius
        self.mass = mass
        self.spin = spin
        self.endmass = endmass

        Jz = radius**2 / 2
        Jxy = (3*radius**2 + length**2) / 12
        inertia = mass * np.diag([Jxy, Jxy, Jz])

        self.bearing = Hinge('bearing', [0,0,1])
        self.pivot   = Hinge('pivot',   [0,1,0])
        self.axis    = Hinge('axis',    [0,0,1])
        self.body    = RigidBody('body', mass, inertia)
        self.offset  = RigidConnection('offset', [0,0,length/2])
        self.endbody = RigidBody('end', endmass)

        self.bearing.add_leaf(self.pivot)
        self.pivot.add_leaf(self.axis)
        self.axis.add_leaf(self.body)
        self.pivot.add_leaf(self.offset)
        self.offset.add_leaf(self.endbody)
        self.system = System(self.bearing)

        # Prescribed DOF accelerations
        self.system.prescribe(self.axis.istrain, 0.0) # constant rotational speed

    def simulate(self, xpivot=0.0, vpivot=0.0, t1=1.5, dt=0.01):
        # reset
        self.system.q [:] = 0.0
        self.system.qd[:] = 0.0

        # initial conditions
        self.system.q [self.pivot.istrain[0]] = xpivot # initial elevation
        self.system.qd[self.pivot.istrain[0]] = vpivot # initial elevation spd
        self.system.qd[self.axis .istrain[0]] = self.spin # initial rotation speed

        # simulate
        self.t = np.arange(0, t1, dt)
        self.y = solve_system(self.system, self.t, outputs='vels')

    def ani(self, vs=1):
        l = self.length * 1.1
        return dynvis.anim(self.system, self.t, self.y,
                           (0,vs), (-l,l), (-l,l), velocities=False)

class BeamGyroscope(Gyroscope):
    def __init__(self, length, radius, mass, spin, endmass):
        self.length = length
        self.radius = radius
        self.mass = mass
        self.spin = spin
        self.endmass = endmass
        Jx = mass * radius**2 / 2

        self.bearing = Hinge('bearing', [0,0,1])
        self.pivot   = Hinge('pivot',   [0,1,0])
        self.offset  = RigidConnection('offset', [-length/2,0,0])
        self.axis    = Hinge('axis',    [1,0,0])
        self.body    = UniformBeam('body', length, mass/length,
                                   1, 1, 1, Jx=Jx/2) # /2 because added to both ends
        self.endbody = RigidBody('end', endmass)

        self.bearing.add_leaf(self.pivot)
        self.pivot.add_leaf(self.offset)
        self.offset.add_leaf(self.axis)
        self.axis.add_leaf(self.body)
        self.offset.add_leaf(self.endbody)
        self.system = System(self.bearing)

        # Prescribed DOF accelerations
        self.system.prescribe(self.axis.istrain, 0.0) # constant rotational speed
        self.system.prescribe(self.body.istrain, 0.0) # rigid beam

class ModalGyroscope(Gyroscope):
    def __init__(self, length, radius, mass, spin, endmass):
        self.length = length
        self.radius = radius
        self.mass = mass
        self.spin = spin
        self.endmass = endmass

        x = np.linspace(-length/2, length/2)
        modes = linearisation.ModalRepresentation(
            x             =x,
            shapes        =np.zeros((len(x),3,0)),
            rotations     =np.zeros((len(x),3,0)),
            density       =np.ones_like(x) * mass/length,
            mass_axis     =np.zeros((len(x),2)),
            section_inertia=np.ones_like(x) * radius**2 / 4,
            freqs         =np.zeros((0,)),
        )

        self.bearing = Hinge('bearing', [0,0,1])
        self.pivot   = Hinge('pivot',   [0,1,0])
        self.axis    = Hinge('axis',    [0,0,1], rotmat_y(-np.pi/2))
        self.body    = ModalElement('body', modes)
        self.offset  = RigidConnection('offset', [0,0,length/2])
        self.endbody = RigidBody('end', endmass)

        self.bearing.add_leaf(self.pivot)
        self.pivot.add_leaf(self.axis)
        self.axis.add_leaf(self.body)
        self.pivot.add_leaf(self.offset)
        self.offset.add_leaf(self.endbody)
        self.system = System(self.bearing)

        # Prescribed DOF accelerations
        self.system.prescribe(self.axis.istrain, 0.0) # constant rotational speed

# Create 3 different models
length = 3.0
radius = 1.0
mass = 100.0
spin = 20.0
endmass = 0.5

#bg = BeamGyroscope(length, radius, mass, spin, endmass)
#mg = ModalGyroscope(length, radius, mass, spin, endmass)
#gg = Gyroscope(length, radius, mass, spin, endmass)

elevation = np.radians(20)

# Theory
spin_inertia = mass * radius**2 / 2
cross_inertia = mass * ((3*radius**2 + length**2) / 12)
torque = endmass * gravity * length/2 * np.sin(elevation)
precession = torque / (spin_inertia * spin * np.sin(elevation))
nutation = spin_inertia * spin / (cross_inertia * np.cos(elevation))

prec2 = (spin*spin_inertia*np.sin(elevation) - np.sqrt(
            (spin*spin_inertia*np.sin(elevation))**2 - \
            4*(cross_inertia-spin_inertia)*np.sin(elevation)*np.cos(elevation)*torque
        )) / \
        (2*(cross_inertia-spin_inertia)*np.sin(elevation)*np.cos(elevation))

# Linearise?
#bl = linearisation.LinearisedSystem(bg.system)
#ml = linearisation.LinearisedSystem(mg.system)
#gl = linearisation.LinearisedSystem(gg.system)

def test():
    # Run simulations
    print '--------------'
    print 'GYROSCOPE TEST'
    print '--------------'
    gg.simulate(elevation)
    mg.simulate(elevation)
    bg.simulate(elevation)
    print 'done.\n\n'

    print "Comparing to RigidBody results. Should have some theory too..."
    print "ModalElement: ", np.allclose(gg.y, mg.y, atol=1e-4) and "ok" or "FAIL"
    print "UniformBeam:  ", np.allclose(gg.y, bg.y, atol=1e-4) and "ok" or "FAIL"

def plotresults(gyro, title=None, velocity=False):
    # Theory
    nutamp = 0.0001
    aznut = nutamp*nutation*np.cos(nutation*gyro.t)
    elnut = nutamp*nutation*np.sin(nutation*gyro.t)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    
    if velocity:
        ax1.plot(gyro.t, gyro.y[:,3], 'b',
                 gyro.t, np.ones_like(gyro.t)*precession + aznut, 'b:')
        ax1.set_ylabel('Azimuth rate (rad/s)')
        ax2.plot(gyro.t, -gyro.y[:,4], 'r',
                 gyro.t, -elnut, 'r--')
        ax2.set_ylabel('Elevation rate (rad/s)')
    else:
        ax1.plot(gyro.t, np.degrees(gyro.y[:,0] % (2*np.pi)), 'b',
                 gyro.t, np.degrees((gyro.t*precession + aznut) % (2*np.pi)), 'b:')
        #ax1.set_ylim((0,360))            
        ax1.set_ylabel('Azimuth / deg')
        ax2.plot(gyro.t, np.degrees(gyro.y[:,1]), 'r',
                 gyro.t, np.degrees(elevation-elnut-nutamp),'r--')
        #ax2.set_ylim((80,90))
        ax2.set_ylabel('Elevation / deg')
    
    ax1.set_xlabel('Time / s')
    if title is not None:
        ax1.set_title(title)


def showplots(velocity=False):
    plotresults(gg, 'RigidBody gyroscope 3m x 1m, 100kg, spinning at 20 rad/s', velocity)
    plotresults(mg, 'ModalElement gyroscope 3m x 1m, 100kg, spinning at 20 rad/s', velocity)
    plotresults(bg, 'UniformBeam gyroscope 3m x 1m, 100kg, spinning at 20 rad/s', velocity)
    plt.show()

if __name__ == '__main__':
    pass
    #test()