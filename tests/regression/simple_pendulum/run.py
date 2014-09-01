#!/usr/bin/env python

from __future__ import division
import numpy as np

import mbwind
from mbwind import (System, RigidBody, Hinge, RigidConnection,
                    Integrator)

# Parameters
mass = 10.0                     # kg
length = 2.5                    # m
natfreq = 1                     # Hz
damping_factor = 0.05           # .


def print_result(result):
    print("\t".join(['t', 'x', 'v', 'a']))
    t = result.t
    y = result.y
    for i in range(len(y[0])):
        print("\t".join(['%.2f' % t[i]] + ['% .8e' % y[j][i, 0]
                                           for j in range(len(y))]))


def build_system():
    # Calculate inertia, stiffness and damping
    I2 = mass * length**2
    k = (2*np.pi*natfreq)**2 * I2
    c = 2 * damping_factor * I2 * (2*np.pi*natfreq)

    # Build model
    hinge = Hinge('hinge', [0,0,1])
    hinge.stiffness = k
    hinge.damping = c
    link = RigidConnection('link', [length, 0, 0])
    body = RigidBody('body', mass)

    system = System()
    system.add_leaf(hinge)
    hinge.add_leaf(link)
    link.add_leaf(body)
    system.setup()

    return system


def simulate(system):
    # Set initial conditions
    system.q[system.elements['hinge'].istrain][0] = 0.3

    # Integrate
    integ = Integrator(system, ('pos','vel','acc'))
    t, y = integ.integrate(5, 0.01, nprint=None)

    return integ


system = build_system()
result = simulate(system)
print_result(result)
