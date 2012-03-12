# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:24:13 2012

@author: Rick Lupton
"""

from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def show(system, tt, yy, tvals):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d', xlabel='x', ylabel='y', zlabel='z')
    ax.plot([1,0,0,0,0],[0,0,1,0,0],[0,0,0,0,1], 'k-')
    ax.set_aspect(1,'datalim')
    ax.hold(True)
    N = yy.shape[1]/2
    for t in tvals:
        i = np.nonzero(t > tt)
        if not len(i) > 0: break
        system.q [system.iDOF] = yy[i[0],:N]
        system.qd[system.iDOF] = yy[i[0],N:]
        system.update(t, False)
        system.first_element.plot_chain(ax)

def anim(system, tt, yy, vs=(0,1), lim1=None, lim2=None):
    fig = plt.figure()
    fig.set_size_inches(10,10,forward=True)
    ax = fig.add_subplot(111, aspect=1, xlim=lim1,ylim=lim2)
    ax.grid()

    lines = []
    for el in system.iter_elements():
        ellines = [ax.plot([], [], **opt)[0] for opt in el.shape_plot_options]
        lines.append( (el,ellines) )
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    ax.set_xlabel('XYZ'[vs[0]])
    ax.set_ylabel('XYZ'[vs[1]])

    def init():
        for el,ellines in lines:
            for line in ellines:
                line.set_data([], [])
        time_text.set_text('')
        return [line for line in ellines for el,ellines in lines] + [time_text]

    N = yy.shape[1]/2
    def animate(i):
        system.q [system.iDOF] = yy[i,:N]
        system.qd[system.iDOF] = yy[i,N:]
        system.update(tt[i], False)

        for el,ellines in lines:
            linedata = el.shape()
            for data,line in zip(linedata,ellines):
                line.set_data(data[:,vs[0]], data[:,vs[1]])
        time_text.set_text(time_template%tt[i])

        return [line for line in ellines for el,ellines in lines] + [time_text]

    ani = animation.FuncAnimation(fig, animate, np.arange(1, yy.shape[0]),
        interval=tt[1]-tt[0]*1000*1, blit=False, init_func=init, repeat=False)
    return ani
