# -*- coding: utf-8 -*-
"""
Created on Tue 3 Apr 2012

Test modelling of a gyroscope and precession/nutation

@author: Rick Lupton
"""

from __future__ import division
import sys
if '..' not in sys.path: sys.path.insert(0,'..')

import numpy as np
from dynamics import System, Hinge, RigidBody, solve_system
import dynvis

import matplotlib.pylab as plt

class Gyroscope(object):
    def __init__(self, length, radius, mass):
        self.length = length
        self.radius = radius
        self.mass = mass

        Jx = radius**2 / 2
        Jyz = (3*radius**2 + length**2) / 12
        inertia = mass * np.diag([Jx, Jyz, Jyz])

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
        ax2.set_ylim((-10,10))
        ax2.set_ylabel('Elevation / deg')



gyro = Gyroscope(3.0, 1.0, 100.0)
system = gyro.system
