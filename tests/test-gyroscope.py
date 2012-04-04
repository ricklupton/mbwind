# -*- coding: utf-8 -*-
"""
Created on Tue 3 Apr 2012

Test modelling of a gyroscope and precession/nutation.

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

from dynamics import (System, Hinge, UniformBeam, RigidBody,
                      ModalElement, solve_system)
import dynvis
import linearisation

import matplotlib.pylab as plt

class Gyroscope(object):
    def __init__(self, length, radius, mass):
        self.length = length
        self.radius = radius
        self.mass = mass

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
        self.system.prescribe(self.axis.istrain, 0.0) # constant rotational speed

    def simulate(self, speed=10, xpivot=0.0, vpivot=0.0, t1=10, dt=0.05):
        # reset
        self.system.q [:] = 0.0
        self.system.qd[:] = 0.0

        # initial conditions
        self.system.q [self.pivot.istrain[0]] = xpivot # initial elevation
        self.system.qd[self.pivot.istrain[0]] = vpivot # initial elevation spd
        self.system.qd[self.axis .istrain[0]] = speed  # initial rotation speed

        # simulate
        self.t = np.arange(0, t1, dt)
        self.y = solve_system(self.system, self.t)

    def ani(self, vs=1):
        l = self.length * 1.1
        return dynvis.anim(self.system, self.t, self.y,
                           (0,vs), (-l,l), (-l,l), velocities=False)

    def plot(self, title=None):
        angles = np.degrees(self.y % (2*np.pi))

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(self.t, angles[:,0], 'b')
        ax1.set_ylim((0,360))
        ax1.set_xlabel('Time / s')
        ax1.set_ylabel('Azimuth / deg')
        if title is not None:
            ax1.set_title(title)

        ax2 = ax1.twinx()
        ax2.plot(self.t, -angles[:,1], 'r')
        ax2.set_ylim((-90,10))
        ax2.set_ylabel('Elevation / deg')

class BeamGyroscope(Gyroscope):
    def __init__(self, length, radius, mass):
        self.length = length
        self.radius = radius
        self.mass = mass
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
        self.system.prescribe(self.axis.istrain, 0.0) # constant rotational speed
        self.system.prescribe(self.body.istrain, 0.0) # rigid beam

class ModalGyroscope(Gyroscope):
    def __init__(self, length, radius, mass):
        self.length = length
        self.radius = radius
        self.mass = mass

        x = np.linspace(0, length)
        modes = linearisation.ModalRepresentation(
            x             =x,
            shapes        =np.zeros((len(x),3,0)),
            rotations     =np.zeros((len(x),3,0)),
            freqs         =np.zeros((0,)),
            density       =np.ones_like(x) * mass/length,
            gyration_radii=np.ones_like(x) * radius**2 / 4,
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
        self.system.prescribe(self.axis.istrain, 0.0) # constant rotational speed

# Create 3 different models
bg = BeamGyroscope(3.0, 1.0, 100.0)
mg = ModalGyroscope(3.0, 1.0, 100.0)
gg = Gyroscope(3.0, 1.0, 100.0)

# Want to test linearisation too?
#bl = linearisation.LinearisedSystem(bg.system)
#ml = linearisation.LinearisedSystem(mg.system)
#gl = linearisation.LinearisedSystem(gg.system)

def test():
    # Run simulations
    print '--------------'
    print 'GYROSCOPE TEST'
    print '--------------'
    gg.simulate()
    mg.simulate()
    bg.simulate()
    print 'done.\n\n'

    print "Comparing to RigidBody results. Should have some theory too..."
    print "ModalElement: ", np.allclose(gg.y, mg.y, atol=1e-3) and "ok" or "FAIL"
    print "UniformBeam:  ", np.allclose(gg.y, bg.y, atol=1e-3) and "ok" or "FAIL"

def showplots():
    gg.plot('RigidBody gyroscope 3m x 1m, 100kg, spinning at 10 rad/s')
    mg.plot('ModalElement gyroscope 3m x 1m, 100kg, spinning at 10 rad/s')
    bg.plot('UniformBeam gyroscope 3m x 1m, 100kg, spinning at 10 rad/s')
    plt.show()

if __name__ == '__main__':
    test()