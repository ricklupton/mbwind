# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:20:19 2012

Test behaviour of a modal beam element with a distal node

@author: Rick Lupton
"""

from __future__ import division

import numpy as np
from numpy import pi, array, dot
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

import dynamics
from dynamics import (ModalElement, System, Integrator, RigidBody)
from blade import Tower

# Options
dynamics.OPT_GRAVITY = True
dynamics.OPT_GEOMETRIC_STIFFNESS = False

# Create model
bladed_file = r'C:\Users\Rick Lupton\Dropbox\phd\Bladed\Models\OC3-Hywind_SparBuoy_NREL5MW.prj'
print "Loading modes from '%s'..." % bladed_file
towerdef = Tower(bladed_file)
modes = towerdef.modal_rep()

endmass = 100000
endinertia = 100

el1 = ModalElement('el', modes, distal=False)
system1 = System(el1)

el2 = ModalElement('el', modes, distal=True)
body = RigidBody('body', endmass, inertia=endinertia*np.eye(3))
el2.add_leaf(body)
system2 = System(el2)

integ1 = Integrator(system1)
integ1.add_output(el1.output_positions())
integ2 = Integrator(system2, outputs=('pos','vel','acc'))
integ2.add_output(el2.output_positions())
integ2.add_output(dynamics.NodeOutput(body.iprox))
integ2.add_output(dynamics.NodeOutput(body.iprox, deriv=2))
integ2.add_output(dynamics.LoadOutput(body.iprox))
integ2.add_output(dynamics.StrainOutput(el2.imult))

if False:
    t,y1 = integ1.integrate(20, 0.05)
    t,y2 = integ2.integrate(20, 0.05)

def p():
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(t, y2[3][:,2], 'b-', label='With end mass')
    ax.plot(t, y1[1][:,2], 'k--', label='Just beam')
    ax.set_ylabel('Z tip defl')
    
    ax = fig.add_subplot(312)
    ax.plot(t, y2[0][:,1], 'b-', label='With end mass')
    ax.plot(t, y1[0][:,1], 'k--', label='Just beam')
    ax.set_ylabel('Translation attachment mode')
    
    ax = fig.add_subplot(313)
    ax.plot(t, y2[0][:,6], 'b-', label='With end mass')
    ax.plot(t, y1[0][:,6], 'k--', label='Just beam')
    ax.set_ylabel('Normal mode')