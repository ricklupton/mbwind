# -*- coding: utf-8 -*-
"""
Created on Tue 20 Mar 2012

Model a pendulum hinged freely to the edge of a disk which spins at constant speed.

@author: Rick Lupton
"""

from __future__ import division
import numpy as np
import dynamics
from dynamics import *
import dynvis
from linearisation import ModalRepresentation

dynamics.gravity = 0

class RotatingBeam(object):
    def __init__(self, r, l, mass):
        self.r = r
        self.l = l
        
        # Mode shapes that represent the hinged rigid beam
        x = np.linspace(0,l,40)
        shapes = np.r_['2,3,0', np.c_[0*x, x/l, 0*x], np.c_[0*x, 0*x, x/l] ]
        rotations = np.r_['2,3,0', np.c_[0*x, 0*x+1/l, 0*x], np.c_[0*x, 0*x, 0*x+1/l] ]
        density = ones_like(x) * mass / l
        freqs = zeros(2)
        self.modes = ModalRepresentation(x, shapes, rotations, density, freqs)

        self.bearing = Hinge('bearing', [0,0,1])
        self.disc = RigidConnection('disc', [r,0,0])
        self.beam = ModalElement('beam', self.modes)

        self.bearing.add_leaf(self.disc)
        self.disc.add_leaf(self.beam)
        self.system = System(self.bearing)

        #hinge_ip.stiffness = 1e3
        #hinge_oop.stiffness = 1e3

        # Prescribed DOF accelerations
        self.system.prescribe(self.beam.istrain, acc=0.0)    # rigid beam
        self.system.prescribe(self.bearing.istrain, acc=0.0) # constant rotational speed

    def init(self, rotspeed, ip0, oop0):
        # reset
        self.system.q [:] = 0.0
        self.system.qd[:] = 0.0

        # initial conditions
        self.system.qd[self.beam.istrain] = [ip0, oop0]    # initial OOP angle
        
        # prescribed values
        self.system.prescribe(self.bearing.istrain, vel=rotspeed, acc=0.0) # constant rotational speed
        
        # setup integrator
        self.integ = Integrator(self.system, ('pos','vel'))
        self.integ.add_output(dynamics.LoadOutput(self.beam.iprox, local=True))
        
    def simulate(self, t1, dt=0.05):
        self.t, self.y = self.integ.integrate(t1, dt)
        return self.t, self.y

def ani_xy(s,t,y):
    return dynvis.anim(s, t, y, (0,1), (-3,3), (-3,3), velocities=False)

def ani_xz(s,t,y):
    return dynvis.anim(s, t, y, (0,2), (-3,3), (-3,3), velocities=False)

def measure_period(t,y):
    t_cross = t[np.nonzero(np.diff(y>=0))]
    print t_cross
    if len(t_cross) > 2: period = np.diff(t_cross[-3::2])[0]
    elif len(t_cross) == 2: period = 2*np.diff(t_cross)[0]
    else: period = np.nan
    return period

rdisc = 1.0
lpend = 1.0
mass = 100

rb = RotatingBeam(rdisc,lpend,mass)
system = rb.system

#rb.init(pi, 0.0, 0.1)
#t,y=rb.simulate(20)

def test(r,l,m,speeds):
    PHI = 1
    PSI = 2

    rb = RotatingBeam(r, l, m)
    # results
    ip = []
    oop = []
    for Omega in speeds:
        rotor_period = 2*pi/Omega

        # in-plane
        rb.init(Omega, 0.4, 0.0)
        t,y = rb.simulate(rotor_period*1.03, dt=rotor_period/200)
        ip.append(2*pi/measure_period(t,y[:,PHI]))

        # out-of-plane
        rb.init(Omega, 0.0, 0.4)
        t,y = rb.simulate(rotor_period*1.03, dt=rotor_period/200)
        oop.append(2*pi/measure_period(t,y[:,PSI]))

    return array([ip,oop]), t,y

speeds = array([0.5,1.0,2.0,5.0])
#freqs,t,y = test(rdisc,lpend,mass,speeds)
gamma = 3*rdisc/lpend**2

def test_geom(m,speed,radii,lengths):
    PHI = 1
    PSI = 2

    rotor_period = 2*pi/speed

    # results
    ip = []
    oop = []
    for r,l in zip(radii,lengths):
        # in-plane
        expected_w = speed*np.sqrt(r/l)
        sim_period = (expected_w>0) and 2*pi/expected_w or rotor_period
        print 'IP: ',expected_w
        rb = RotatingBeam(r, l, m)
        rb.init(speed, 0.4, 0.0)
        t1,y1 = rb.simulate(sim_period*2.5, dt=2.5*sim_period/300)
        ip.append(2*pi/measure_period(t1,y1[PHI][:,0]))

        # out-of-plane
        expected_w = speed*np.sqrt(1+r/l)
        sim_period = 2*pi/expected_w
        print 'OOP:',expected_w
        rb = RotatingBeam(r, l, m)
        rb.init(speed, 0.0, 0.4)
        t2,y2 = rb.simulate(sim_period*2.5, dt=2.5*sim_period/300)
        oop.append(2*pi/measure_period(t2,y2[PSI][:,0]))

    return array([ip,oop]), t1,y1,t2,y2

if True:
    rads = array([0.0,0.1,0.3,0.5,1.0,1.0,2.0,3.0,9.0])
    lens = array([1.0,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5])
    gamma = rads/lens
    lambd = gamma / (gamma+1)
    rotspeed = pi
    freqs,t1,y1,t2,y2 = test_geom(mass,pi,rads,lens)
    x=np.linspace(0,0.96)
    
    import matplotlib.pyplot as plt
    plt.plot(lambd,freqs[0]/rotspeed,'b-x',lambd,freqs[1]/rotspeed,'g-x',
             x,sqrt(x/(1-x)),'b--',x,sqrt(1/(1-x)),'g--')
    plt.xlabel('$\lambda$ (disc inertia / total inertia)')
    plt.ylabel('Natural frequencies normalised by rotor frequency')
    plt.legend(('Simulated IP','Simulated OOP','Theory IP','Theory OOP'),loc='nw')
    plt.title('Disc+rigid beam rotating at %.2f rad/s' % rotspeed)

def plot_defl(t,y):
    ip  = [b+i for i in (1,5) for b in (8,14,20)]
    oop = [b+i for i in (2,4) for b in (8,14,20)]
    print ip

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(t,y[:,ip])
    #ax1.legend(('B1 tr','B1 rot','B2 tr','B2 rot','B3 tr','B3 rot'))
    ax1.set_title('In plane')
    ax1.set_xlabel('Time / s')
    ax1.set_ylabel('Strain')

    ax2 = fig.add_subplot(122)
    ax2.plot(t,y[:,oop])
    ax2.legend(('B1 tr','B2 tr','B3 tr','B1 rot','B2 rot','B3 rot'))
    ax2.set_title('Out of plane')
    ax2.set_xlabel('Time / s')
