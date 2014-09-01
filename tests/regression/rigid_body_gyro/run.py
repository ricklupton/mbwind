#!/usr/bin/env python

from __future__ import division
import numpy as np

from mbwind import (System, RigidBody, Hinge, Integrator)


class Gyroscope:
    def __init__(self, length, radius, mass, spin):
        self.length = length
        self.radius = radius
        self.mass = mass
        self.spin = spin

        self.bearing = Hinge('bearing', [0, 0, 1])
        self.pivot = Hinge('pivot', [0, 1, 0])
        self.axis = Hinge('axis', [1, 0, 0])
        self.body = self.build_body()

        self.pivot.damping = 200

        self.system = System(gravity=9.81)
        self.system.add_leaf(self.bearing)
        self.bearing.add_leaf(self.pivot)
        self.pivot.add_leaf(self.axis)
        self.axis.add_leaf(self.body)
        self.system.setup()

        # Prescribed DOF accelerations: constant rotational speed
        self.system.prescribe(self.axis, vel=self.spin, acc=0.0)

    def build_body(self):
        Jx = self.radius**2 / 2
        Jyz = (3*self.radius**2 + self.length**2) / 12
        Jyz_0 = Jyz + (self.length/2)**2  # parallel axis theorem
        inertia = self.mass * np.diag([Jx, Jyz_0, Jyz_0])
        return RigidBody('body', self.mass, inertia, [self.length/2, 0, 0])

    def simulate(self, xpivot=0.0, vpivot=0.0, t1=10, dt=0.05):
        # reset
        self.system.q[:] = 0.0
        self.system.qd[:] = 0.0

        # initial conditions
        self.system.q[self.pivot.istrain][0] = xpivot  # initial elevation
        #self.system.qd[self.pivot.istrain][0] = vpivot # initial elevation spd

        # simulate
        integ = Integrator(self.system, ('pos', 'vel'))
        integ.integrate(t1, dt, nprint=None)
        return integ


def print_result(result):
    print("\t".join(['t', 'phi', 'theta', 'psi']))
    t = result.t
    y = result.y
    for i in range(len(y[0])):
        print("\t".join(['%.2f' % t[i]] + ['% .8e' % y[j][i, 0]
                                           for j in range(3)]))


gyro = Gyroscope(length=3.0,
                 radius=1.0,
                 mass=100.0,
                 spin=30.0)
result = gyro.simulate()
print_result(result)
