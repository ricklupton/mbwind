# -*- coding: utf-8 -*-
"""
Created on Wed 28 Mar 2012

Test linearisation of a rigid 3-bladed rotor which can bounce vertically

@author: Rick Lupton
"""

from __future__ import division
import sys
if '..' not in sys.path: sys.path.insert(0,'..')

import numpy as np
import scipy.linalg

import dynamics
from dynamics import *
from linearisation import LinearisedSystem
import dynvis

dynamics.gravity = 0

tower_height = 90.0
overhang = 0.8
r_root = 0.5
blade_length = 60.0

EIy = 1000e6
EIz = 1000e6

Ry = rotmat_y(-pi/2)
Rhb1 = rotmat_z(0 * 2*pi/3)
Rhb2 = rotmat_z(1 * 2*pi/3)
Rhb3 = rotmat_z(2 * 2*pi/3)

slider = PrismaticJoint('slider', [0,0,1], Ry)
beam = EulerBeam('beam', tower_height, 3000, 1, 1, 1)
hinge = Hinge('hinge',  [0,0,1])
hb1 = RigidConnection('hb1', r_root*np.dot(Rhb1,[1,0,0]), Rhb1)
hb2 = RigidConnection('hb2', r_root*np.dot(Rhb2,[1,0,0]), Rhb2)
hb3 = RigidConnection('hb3', r_root*np.dot(Rhb3,[1,0,0]), Rhb3)
b1 = EulerBeam('b1', blade_length, 250, 1000e6, EIy, EIz)
b2 = EulerBeam('b2', blade_length, 250, 1000e6, EIy, EIz)
b3 = EulerBeam('b3', blade_length, 250, 1000e6, EIy, EIz)

slider.add_leaf(beam)
beam.add_leaf(hinge)
hinge.add_leaf(hb1)
hinge.add_leaf(hb2)
hinge.add_leaf(hb3)
hb1.add_leaf(b1)
hb2.add_leaf(b2)
hb3.add_leaf(b3)
system = System(slider)

slider.stiffness = 1e7
        
# Prescribed DOF accelerations
system.prescribe(beam.istrain, 0.0)    # rigid beam
for b in (b1,b2,b3):
    x = b.istrain
    system.prescribe([x[0],x[2],x[3],x[4]], 0.0) # only deflection in y-direction & rotation about z (IP)

linsys = LinearisedSystem(system) # about zeros
W,Vr = scipy.linalg.eig(linsys.K,linsys.M)

def calc_matrices(system, iz, omega, nvals=50, mbc=False):
    f = system.B.shape[0]
    z = zeros(f)
    vals = linspace(0, 2*pi, nvals)
    result = zeros((nvals,f,f))
    for i,v in enumerate(vals):
        z[iz] = v
        linsys = LinearisedSystem(system, z)
        if mbc:
            linsys = linsys.multiblade_transform(1, slice(2,4), slice(4,6), slice(6,8))
        result[i] = linsys.M
    return result

def ani_matrix(mats, vals=None):
    if vals is None: vals = arange(mats.shape[0])
    import matplotlib.animation as animation
    import matplotlib as mpl 
    fig = plt.figure()
    fig.set_size_inches(5,5,forward=True)
    ax = fig.add_subplot(111, aspect=1)
    ax.grid()
    
    norm = mpl.colors.Normalize(vmin=(mats-mats[0]).min(), vmax=(mats-mats[0]).max())
    cmap = mpl.cm.get_cmap('coolwarm', 20) 
    
    def animate(i):
        m = ax.matshow(mats[i]-mats[0], norm=norm, cmap=cmap)
        ax.set_title(vals[i])
        if i == 1: fig.colorbar(m)

    ani = animation.FuncAnimation(fig, animate, range(len(vals)),
        interval=1, blit=False, repeat=False)
    return ani
    
if False:
    mats = calc_matrices(system, 1, 0, mbc=False)
    ani_matrix(mats)

def sim_both(a,b,a1,b1):
    t = np.arange(0, 20, 0.05)
    ylin = linsys.integrate(t, [a,b]+[0]*6, [a1,b1]+[0]*6)
    system.q [system.iFreeDOF[0]] = a
    system.q [system.iFreeDOF[1]] = b
    system.qd[system.iFreeDOF[0]] = a1
    system.qd[system.iFreeDOF[1]] = b1
    y = solve_system(system,t)
    
    # pick out interesting strains
    y = y[:,[0,7,9,13,15,19,21,25]]
    ylin = ylin[:,:8]
    ax = plt.figure().add_subplot(111)
    #ax.set_color_cycle(['b','r'])
    ax.plot(t,ylin,':',t,y,'--')
    return t,y,ylin

def plot_three(t,ynl,ylin,ymbc):
    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.set_color_cycle(['b','r'])
    ax.plot(t,ylin[:,:2],':',t,ymbc[:,:2],'--',t,ynl[:,:2],'-')
    ax = fig.add_subplot(312)
    ax.set_color_cycle(['r','g','b'])
    ax.plot(t,ylin[:,2::2],':',t,ymbc[:,2::2],'--',t,ynl[:,2::2],'-')
    ax = fig.add_subplot(313)
    ax.set_color_cycle(['r','g','b'])
    ax.plot(t,ylin[:,3::2],':',t,ymbc[:,3::2],'--',t,ynl[:,3::2],'-')

def sim_three(system,a,b,a1,b1):
    t = np.arange(0, 20, 0.05)
    linsys = LinearisedSystem(system, [a,b]+[0]*6, [a1,b1]+[0]*6)
    mbcsys = linsys.multiblade_transform(1, slice(2,4), slice(4,6), slice(6,8))
    ylin = linsys.integrate(t, [a,b]+[0]*6, [a1,b1]+[0]*6)
    ymbc = mbcsys.integrate(t, [a,b]+[0]*6, [a1,b1]+[0]*6)
    system.q [system.iFreeDOF[0]] = a
    system.q [system.iFreeDOF[1]] = b
    system.qd[system.iFreeDOF[0]] = a1
    system.qd[system.iFreeDOF[1]] = b1
    ynl = solve_system(system,t)
    
    # pick out interesting strains
    ynl = ynl[:,[0,7,9,13,15,19,21,25]]
    ylin = ylin[:,:8]
    ymbc = ymbc[:,:8]
    plot_three(t,ynl,ylin,ymbc)
    return t,ynl,ylin,ymbc
    
def simulate(system, t1, dt=0.05):
    t = np.arange(0, t1, dt)
    y = solve_system(system, t)
    return t,y

def ani_xy(s,t,y):
    return dynvis.anim(s, t, y, (0,1), (-10,10), (-10,10), velocities=False, only_free=True)

def ani_xz(s,t,y):
    return dynvis.anim(s, t, y, (0,2), (-10,10), (-10,20), velocities=False, only_free=True)

def ani_yz(s,t,y):
    return dynvis.anim(s, t, y, (1,2), (-10,10), (-10,20), velocities=False, only_free=True)

