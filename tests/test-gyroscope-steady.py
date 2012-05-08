# -*- coding: utf-8 -*-
"""
Created on Tue 3 Apr 2012

Test modelling of a gyroscope and precession. Aim to avoid nutation.

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

from dynamics import (System, Hinge, UniformBeam, RigidBody, Integrator,
                      ModalElement, gravity)
import dynamics
import dynvis
import linearisation

import matplotlib.pylab as plt

class Gyroscope(object):
    def __init__(self, length, radius, mass, spin):
        self.length = length
        self.radius = radius
        self.mass = mass
        self.spin = spin

        Jx = radius**2 / 2
        Jyz = (3*radius**2 + length**2) / 12
        Jyz_0 = Jyz + (length/2)**2 # parallel axis theorem
        inertia = mass * np.diag([Jx, Jyz_0, Jyz_0])

        self.bearing = Hinge('bearing', [0,0,1])
        self.pivot   = Hinge('pivot',   [0,1,0])
        self.axis    = Hinge('axis',    [1,0,0])
        self.body    = RigidBody('body', mass, inertia, [length/2, 0, 0])

        self.bearing.add_leaf(self.pivot)
        self.pivot.add_leaf(self.axis)
        self.axis.add_leaf(self.body)
        self.system = System(self.bearing)

        # Prescribed DOF accelerations
        self.system.prescribe(self.axis, acc=0.0) # constant rotational speed

    def simulate(self, xpivot=0.0, vprec=0.0, t1=10, dt=0.05):
        # reset
        self.system.q [:] = 0.0
        self.system.qd[:] = 0.0

        # initial conditions
        self.system.q [self.pivot.istrain][0] = xpivot # initial elevation
        self.system.qd[self.bearing.istrain][0] = vprec # initial elevation spd
        self.system.qd[self.axis .istrain][0] = self.spin # initial rotation speed

        self.integ = Integrator(self.system, ('pos','vel'))
        self.integ.add_output(dynamics.LoadOutput(self.axis.iprox))
        self.integ.add_output(dynamics.LoadOutput(self.pivot.iprox))
        self.integ.add_output(dynamics.LoadOutput(self.bearing.iprox))

        # simulate
        if t1 > 0:
            self.t, self.y = self.integ.integrate(t1, dt)
            for i,lab in enumerate(self.integ.labels()):
                print "%2d  %s" % (i,lab)
            return self.t, self.y
        
    def ani(self, vs=1):
        l = self.length * 1.1
        return dynvis.anim(self.system, self.t, self.y,
                           (0,vs), (-l,l), (-l,l), velocities=False)

class BeamGyroscope(Gyroscope):
    def __init__(self, length, radius, mass, spin):
        self.length = length
        self.radius = radius
        self.mass = mass
        self.spin = spin
        Jx = mass * radius**2 / 2

        self.bearing = Hinge('bearing', [0,0,1])
        self.pivot   = Hinge('pivot',   [0,1,0])
        self.axis    = Hinge('axis',    [1,0,0])
        self.body    = UniformBeam('body', length, mass/length,
                                   1, 1, 1, Jx=Jx/2) # /2 because added to both ends

        self.bearing.add_leaf(self.pivot)
        self.pivot.add_leaf(self.axis)
        self.axis.add_leaf(self.body)
        self.system = System(self.bearing)

        # Prescribed DOF accelerations
        self.system.prescribe(self.axis, acc=0.0) # constant rotational speed
        self.system.prescribe(self.body, acc=0.0) # rigid beam

class ModalGyroscope(Gyroscope):
    def __init__(self, length, radius, mass, spin):
        self.length = length
        self.radius = radius
        self.mass = mass
        self.spin = spin

        x = np.linspace(0, length)
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
        self.axis    = Hinge('axis',    [1,0,0])
        self.body    = ModalElement('body', modes)

        self.bearing.add_leaf(self.pivot)
        self.pivot.add_leaf(self.axis)
        self.axis.add_leaf(self.body)
        self.system = System(self.bearing)

        # Prescribed DOF accelerations
        self.system.prescribe(self.axis, acc=0.0) # constant rotational speed

# Create 3 different models
length = 7.0
radius = 1.0
mass = 100.0
spin = 20.0

bg = BeamGyroscope(length, radius, mass, spin)
mg = ModalGyroscope(length, radius, mass, spin)
gg = Gyroscope(length, radius, mass, spin)

# Theory
spin_inertia = mass * radius**2 / 2
cross_inertia = mass * (3*radius**2 + length**2) / 12
spin_velocity = 10
torque = mass * gravity * length/2
precession = torque / (spin_inertia * spin_velocity)
nutation = cross_inertia * spin_velocity / spin_inertia

# Want to test linearisation too?
#bl = linearisation.LinearisedSystem(bg.system)
#ml = linearisation.LinearisedSystem(mg.system)
#gl = linearisation.LinearisedSystem(gg.system)

def test():
    # Run simulations
    print '--------------'
    print 'GYROSCOPE TEST'
    print '--------------'
    rate = 2 
    gg.simulate(vprec=rate)
    mg.simulate(vprec=rate)
    bg.simulate(vprec=rate)
    print 'done.\n\n'

    atol = 1e-4
    mok = np.allclose(gg.y[0], mg.y[0], atol=atol) and np.allclose(gg.y[1], mg.y[1], atol=atol)
    bok = np.allclose(gg.y[0], bg.y[0], atol=atol) and np.allclose(gg.y[1], bg.y[1], atol=atol)

    print "Comparing to RigidBody results. Should have some theory too..."
    print "ModalElement: ", mok and "ok" or "FAIL"
    print "UniformBeam:  ", bok and "ok" or "FAIL"

def plotresults(gyro, title=None, velocity=False):
    # Theory
    spin_inertia = gyro.mass * gyro.radius**2 / 2
    cross_inertia = gyro.mass * (3*gyro.radius**2 + gyro.length**2) / 12
    spin_velocity = gyro.spin
    torque = gyro.mass * gravity * gyro.length/2
    precession = torque / (spin_inertia * spin_velocity)
    nutation = cross_inertia * spin_velocity / spin_inertia
    nutamp = 0.1

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = ax1.twinx()

    ax1.plot(gyro.t, np.degrees(gyro.y[0] % (2*np.pi)), 'b',
             gyro.t, np.degrees((gyro.t*precession) % (2*np.pi)), 'b:')
    #ax1.set_ylim((0,360))
    ax1.set_ylabel('Azimuth (deg)')
    ax1.yaxis.label.set_color('blue')
    ax2.plot(gyro.t, np.degrees(gyro.y[1]), 'r',
             gyro.t, np.zeros_like(gyro.t),'r--')
    #ax2.set_ylim((15,25))
    ax2.set_ylabel('Elevation (deg)')
    ax2.yaxis.label.set_color('red')

    ax3 = fig.add_subplot(212)
    ax4 = ax3.twinx()
    
    ax3.set_color_cycle(['r','g','b'])
    ax3.plot(gyro.t, gyro.y[6][:,:3], '--')
    ax3.set_ylabel('Base forces [--]')
    ax4.set_color_cycle(['r','g','b'])
    ax4.plot(gyro.t, gyro.y[6][:,3:], '-')
    ax4.set_ylabel('Base moments [-]')
    ax4.legend(('x','y','z'))

    ax3.set_xlabel('Time / s')
    if title is not None:
        ax1.set_title(title)

def showplots(velocity=False):
    gg.plot('RigidBody gyroscope 3m x 1m, 100kg, spinning at 10 rad/s', velocity)
    mg.plot('ModalElement gyroscope 3m x 1m, 100kg, spinning at 10 rad/s', velocity)
    bg.plot('UniformBeam gyroscope 3m x 1m, 100kg, spinning at 10 rad/s', velocity)
    plt.show()

if __name__ == '__main__':
    pass
    #test()