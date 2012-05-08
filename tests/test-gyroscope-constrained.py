# -*- coding: utf-8 -*-
"""
Created on Wed 25 Apr 2012

Test modelling of gyroscopic effects, with the centre of mass of a rotating
body located at the origin. The body can rotate in a vertical plane. A small
mass on an offset provides a torque in the vertical plane.

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

import dynamics
from dynamics import (System, Hinge, UniformBeam, RigidBody, RigidConnection,
                      ModalElement, Integrator, gravity, rotmat_y)
import dynvis
import linearisation

import matplotlib.pylab as plt

class Gyroscope(object):
    def __init__(self, length, radius, mass, endmass):
        self.length = length
        self.radius = radius
        self.mass = mass
        self.endmass = endmass

        Jz = radius**2 / 2
        Jxy = (3*radius**2 + length**2) / 12
        inertia = mass * np.diag([Jxy, Jxy, Jz])

        self.A = mass*Jxy + endmass*(length/2)**2
        self.C = mass*Jz

        self.pivot   = Hinge('pivot',   [0,1,0])
        self.axis    = Hinge('axis',    [0,0,1])
        self.body    = RigidBody('body', mass, inertia)
        self.offset  = RigidConnection('offset', [0,0,length/2])
        self.endbody = RigidBody('end', endmass)

        self.pivot.add_leaf(self.axis)
        self.axis.add_leaf(self.body)
        
        self.pivot.add_leaf(self.offset)
        self.offset.add_leaf(self.endbody)

        self.system = System(self.pivot)

        # Prescribed DOF accelerations
        self.system.prescribe(self.axis, acc=0.0) # constant rotational speed

    def applied_torque(self, s):
        # applied torque about y axis, Q2
        elevation = s.q[self.pivot.istrain][0]
        return self.endmass * dynamics.gravity * self.length/2 * np.sin(elevation)
    
    def thetadotdot(self, s):
        # acceleration about y axis, thetadotdot
        Q2 = self.applied_torque(s)
        return Q2 / self.A
    
    def gyro_torque(self, s):
        # gyroscopic sideways torque, Q1
        spin = s.qd[self.axis.istrain][0] # spin speed
        thetadot = s.qd[self.pivot.istrain][0]
        return self.C * spin * thetadot
        
    def simulate(self, xpivot=0.0, spin=10.0, t1=1.5, dt=0.01):
        # reset
        self.system.q [:] = 0.0
        self.system.qd[:] = 0.0

        # initial conditions
        self.system.q [self.pivot.istrain][0] = xpivot # initial elevation
        self.system.qd[self.axis .istrain][0] = spin # initial rotation speed

        # setup integrator
        #self.integ = Integrator(self.system, ('pos','vel','acc'))
        self.integ = Integrator(self.system, ())
        for istrain in (self.pivot.istrain, self.axis.istrain):
            self.integ.add_output(dynamics.StrainOutput(istrain, deriv=0))
            self.integ.add_output(dynamics.StrainOutput(istrain, deriv=1))
            self.integ.add_output(dynamics.StrainOutput(istrain, deriv=2))
        self.integ.add_output(dynamics.LoadOutput(self.pivot.idist[0]))
        self.integ.add_output(dynamics.LoadOutput(self.pivot.idist[0], local=True))
        self.integ.add_output(dynamics.LoadOutput(self.body.iprox))
        self.integ.add_output(dynamics.LoadOutput(self.endbody.iprox))
        self.integ.add_output(dynamics.CustomOutput(self.applied_torque, "Q2"))
        self.integ.add_output(dynamics.CustomOutput(self.thetadotdot, "thetadotdot"))
        self.integ.add_output(dynamics.CustomOutput(self.gyro_torque, "Q1"))

        # simulate
        if t1 > 0:
            self.t,self.y = self.integ.integrate(t1, dt)

    def ani(self, vs=1):
        l = self.length * 1.1
        return dynvis.anim(self.system, self.t, self.y,
                           (0,vs), (-l,l), (-l,l), velocities=False, only_free=True)

class LumpedGyroscope(Gyroscope):
    def __init__(self, length, radius, mass, endmass):
        self.length = length
        self.radius = radius
        self.mass = mass
        self.endmass = endmass

        # corrected radius for lumped masses
        mr = radius / np.sqrt(2)
        root3 = np.sqrt(3)
        
        self.A = 3 * (mass/3) * mr**2 / 2 + endmass*(length/2)**2
        self.C = 3 * (mass/3) * mr**2

        self.pivot   = Hinge('pivot',   [0,1,0])
        self.axis    = Hinge('axis',    [0,0,1])
        self.arm1    = RigidConnection('arm1', [mr,0,0])
        self.arm2    = RigidConnection('arm2', [-mr/2, mr*root3/2,0])
        self.arm3    = RigidConnection('arm3', [-mr/2,-mr*root3/2,0])
        self.body    = RigidBody('body',  mass/3)
        self.body2   = RigidBody('body2', mass/3)
        self.body3   = RigidBody('body3', mass/3)
        self.offset  = RigidConnection('offset', [0,0,length/2])
        self.endbody = RigidBody('end', endmass)

        self.pivot.add_leaf(self.axis)
        self.axis.add_leaf(self.arm1)
        self.axis.add_leaf(self.arm2)
        self.axis.add_leaf(self.arm3)
        self.arm1.add_leaf(self.body)
        self.arm2.add_leaf(self.body2)
        self.arm3.add_leaf(self.body3)
        
        self.pivot.add_leaf(self.offset)
        self.offset.add_leaf(self.endbody)
        self.system = System(self.pivot)

        # Prescribed DOF accelerations
        self.system.prescribe(self.axis, acc=0.0) # constant rotational speed

class BeamGyroscope(Gyroscope):
    def __init__(self, length, radius, mass, endmass):
        self.length = length
        self.radius = radius
        self.mass = mass
        self.endmass = endmass
        
        Jx = mass * radius**2 / 2        
        Jyz = mass*(3*radius**2 + length**2) / 12        
        self.A = Jyz + endmass*(length/2)**2
        self.C = Jx

        self.pivot   = Hinge('pivot',   [0,1,0])
        self.offset  = RigidConnection('offset', [0,0,-length/2])
        self.axis    = Hinge('axis',    [0,0,1], rotmat_y(-np.pi/2))
        self.body    = UniformBeam('body', length, mass/length,
                                   1, 1, 1, Jx=Jx/2) # /2 because added to both ends
        self.endoffset = RigidConnection('endoffset', [0,0,length/2])
        self.endbody = RigidBody('end', endmass)

        self.pivot.add_leaf(self.offset)
        self.offset.add_leaf(self.axis)
        self.axis.add_leaf(self.body)
        
        self.pivot.add_leaf(self.endoffset)
        self.endoffset.add_leaf(self.endbody)
        self.system = System(self.pivot)

        # Prescribed DOF accelerations
        self.system.prescribe(self.axis, acc=0.0) # constant rotational speed
        self.system.prescribe(self.body, acc=0.0) # rigid beam

class ModalGyroscope(Gyroscope):
    def __init__(self, length, radius, mass, endmass):
        self.length = length
        self.radius = radius
        self.mass = mass
        self.endmass = endmass

        Jx = mass * radius**2 / 2        
        Jyz = mass*(3*radius**2 + length**2) / 12
        self.A = Jyz + endmass*(length/2)**2
        self.C = Jx

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

        self.pivot   = Hinge('pivot',   [0,1,0])
        self.axis    = Hinge('axis',    [0,0,1], rotmat_y(-np.pi/2))
        self.body    = ModalElement('body', modes)
        self.offset  = RigidConnection('offset', [0,0,length/2])
        self.endbody = RigidBody('end', endmass)

        self.pivot.add_leaf(self.axis)
        self.axis.add_leaf(self.body)
        self.pivot.add_leaf(self.offset)
        self.offset.add_leaf(self.endbody)
        self.system = System(self.pivot)

        # Prescribed DOF accelerations
        self.system.prescribe(self.axis, acc=0.0) # constant rotational speed

# Create 3 different models
length = 3.0
radius = 1.0
mass = 100.0
spin = 20.0
endmass = 0.5

#bg = BeamGyroscope(length, radius, mass, endmass)
#mg = ModalGyroscope(length, radius, mass, endmass)
#gg = Gyroscope(length, radius, mass, endmass)
#lg = LumpedGyroscope(length, radius, mass, endmass)

elevation = np.radians(20)

#gg.simulate(elevation, 0.00245, t1=5)

def cmpgyro(y1, y2, rtol=1e-5, atol=1e-8):
    for a,b in zip(y1,y2):
        if not np.allclose(a,b,rtol,atol): return False
    return True

def test():
    # Run simulations
    print '--------------'
    print 'GYROSCOPE TEST'
    print '--------------'
    gg.simulate(elevation, spin)
    mg.simulate(elevation, spin)
    #bg.simulate(elevation, spin)
    lg.simulate(elevation, spin)
    print 'done.\n\n'

    print "Comparing to RigidBody results:"
    print "Modal element: ", cmpgyro(gg.y, mg.y, atol=1e-3) and "ok" or "FAIL"
    print "Lumped masses: ", cmpgyro(gg.y, lg.y, atol=1e-3) and "ok" or "FAIL"
    #print "UniformBeam:  ", np.allclose(gg.y, bg.y, atol=1e-4) and "ok" or "FAIL"

def p(gyro):
    t,y = gyro.t, gyro.y
    fig = plt.figure()
    
    ax = fig.add_subplot(221)
    ax.plot(t, -y[6][:,4], t, y[10], 'k--') ### NB applied torque is -ve reaction torque
    ax.set_title('Applied torque $Q_2$')
    
    ax = fig.add_subplot(222)
    ax.plot(t, y[4], t, y[11], 'k--')
    ax.set_title(r'$\ddot\theta$')
    
    ax = fig.add_subplot(223)
    ax.plot(t, y[7][:,3], t, y[12], 'k--')
    ax.set_title(r'Gyro torque $Q_1$')
    
def p2(gyro1, gyro2):
    t,y1,y2 = gyro1.t, gyro1.y, gyro2.y
    fig = plt.figure()
    
    ax = fig.add_subplot(221)
    ### NB applied torque is -ve reaction torque
    ax.plot(t, -y1[6][4].T, t, -y2[6][4].T, t, y1[10].T, 'k--') 
    ax.set_title('Applied torque $Q_2$')
    
    ax = fig.add_subplot(222)
    ax.plot(t, y1[4].T, t, y2[4].T, t, y1[11].T, 'k--')
    ax.set_title(r'$\ddot\theta$')
    
    ax = fig.add_subplot(223)
    ax.plot(t, y1[7][3].T, t, y2[7][3].T, t, y1[12].T, 'k--')
    ax.set_title(r'Gyro torque $Q_1$')
    
def plotresults(gyro, title=None, velocity=False):
    # Theory

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    
    if velocity:
        ax1.plot(gyro.t, gyro.y[:,3], 'b',
                 gyro.t, np.ones_like(gyro.t)*precession, 'b:')
        ax1.set_ylabel('Azimuth rate (rad/s)')
        ax2.plot(gyro.t, -gyro.y[:,4], 'r',
                 gyro.t, 0*gyro.t, 'r--')
        ax2.set_ylabel('Elevation rate (rad/s)')
    else:
        ax1.plot(gyro.t, np.degrees(gyro.y[:,0] % (2*np.pi)), 'b',
                 gyro.t, np.degrees((gyro.t*precession) % (2*np.pi)), 'b:')
        #ax1.set_ylim((0,360))            
        ax1.set_ylabel('Azimuth / deg')
        ax2.plot(gyro.t, np.degrees(gyro.y[:,1]), 'r',
                 gyro.t, np.ones_like(gyro.t)*np.degrees(elevation),'r--')
        ax2.set_ylim((0,30))
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