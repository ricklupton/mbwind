# -*- coding: utf-8 -*-
"""
Created on Wed 28 Mar 2012

Test mode shapes found from a beam made up of several elements

@author: Rick Lupton
"""

from __future__ import division
import sys
if '..' not in sys.path: sys.path.insert(0,'..')

import numpy as np
from scipy.linalg import eig

import dynamics
from dynamics import System, ModalElement
from linearisation import LinearisedSystem, BeamModalAnalysis
import dynvis

dynamics.gravity = 0

beam_length = 60.0
beam_elements = 15

total_mass = 17e3
density = total_mass/beam_length
EA  = 1000e6
EIy = 1000e6
EIz = 1000e6
GIx = 1000e6
Jx  = 1000

ma = BeamModalAnalysis(beam_length, density, EA, EIy, EIz, GIx, Jx,
                       Nelements=beam_elements, Nmodes=1)

# Compare to modal element
el = ModalElement('el', ma.rep)
system = System(el)
linsys = LinearisedSystem(system)
w,v = eig(linsys.K, linsys.M)

#dynvis.anim_mode(system, vr[:,0], (-5,65), (-5,5), (-5,5))


def calc_matrices(system, iz, vals, mat='M'):
    f = system.B.shape[0]
    z0 = zeros(f)
    result = zeros((len(vals),f,f))
    for i,v in enumerate(vals):
        z0[iz] = v
        linsys = LinearisedSystem(system, z0)
        result[i] = getattr(linsys,mat)
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
    mats = calc_matrices(system, 1, linspace(-1,1))
    ani_matrix(mats)

def sim_both(a,b,a1,b1):
    t = np.arange(0, 20, 0.05)
    ylin = linsys.integrate(t, [a,b,0,0,0], [a1,b1,0,0,0])
    system.q [:] = 0.0
    system.qd[:] = 0.0
    system.q [system.iFreeDOF[0]] = a
    system.q [system.iFreeDOF[1]] = b
    system.qd[system.iFreeDOF[0]] = a1
    system.qd[system.iFreeDOF[1]] = b1
    y = solve_system(system,t)

    # pick out interesting strains
    y = y[:,[0,7,8,15,22]]
    ylin = ylin[:,:5]
    ax = plt.figure().add_subplot(111)
    #ax.set_color_cycle(['b','r'])
    ax.plot(t,ylin,':',t,y,'--')
    return t,y,ylin

def plot_three(t,ynl,ylin,ymbc):
    fig = plt.figure()
    titles = ['Non-lin','Linear','MBC']
    for i,y in enumerate((ynl,ylin,ymbc)):
        ax = fig.add_subplot(2,3,1+i)
        ax.set_color_cycle(['b','r'])
        ax.plot(t,y[:,:2])
        ax.set_title(titles[i])
    for i,y in enumerate((ynl,ylin,ymbc)):
        ax = fig.add_subplot(2,3,4+i)
        ax.plot(t,y[:,2:])
        ax.set_color_cycle(['r','g','b'])

def sim_three(system,linsys,a,b,a1,b1):
    t = np.arange(0, 20, 0.1)
    mbcsys = linsys.multiblade_transform(1, slice(2,3), slice(3,4), slice(4,5))
    ylin = linsys.integrate(t, [a,b,0,0,0], [a1,b1,0,0,0])
    ymbc = mbcsys.integrate(t, [a,b,0,0,0], [a1,b1,0,0,0])
    system.q [system.iFreeDOF[0]] = a
    system.q [system.iFreeDOF[1]] = b
    system.qd[system.iFreeDOF[0]] = a1
    system.qd[system.iFreeDOF[1]] = b1
    ynl = solve_system(system,t)

    # pick out interesting strains
    ynl = ynl[:,[0,7,8,15,22]]
    ylin = ylin[:,:5]
    ymbc = ymbc[:,:5]
    plot_three(t,ynl,ylin,ymbc)
    return t,ynl,ylin,ymbc

def simulate(system, t1, dt=0.05):
    t = np.arange(0, t1, dt)
    y = solve_system(system, t)
    return t,y

def ani_xy(s,t,y):
    return dynvis.anim(s, t, y, (0,1), (-5,65), (-10,10), velocities=False)

def ani_xz(s,t,y):
    return dynvis.anim(s, t, y, (0,2), (-5,65), (-10,20), velocities=False)

def ani_yz(s,t,y):
    return dynvis.anim(s, t, y, (1,2), (-10,10), (-10,20), velocities=False, only_free=True)

