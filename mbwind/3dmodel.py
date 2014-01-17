# -*- coding: utf-8 -*-
"""
Created on Fri May 04 17:56:18 2012

@author: Rick Lupton
"""

from .visual import *
from numpy import *

scene.background = (1,1,1)
scene.foreground = (0.5,0.5,0.6)
scene.ambient = 0.5

axis = curve(pos=[(0,-10,0),(0,10,0)])
label(pos=(0,-10,0), text='Motion about\npitch axis', xoffset=40, space=5,
      color=color.black)

tower = cylinder(pos=vector(0,0,0), radius=2, axis=vector(0,0,60))
nacelle = box(pos=vector(0,0,60), size=(8,5,4))
hub = sphere(pos=vector(-6,0,60), radius=2)
hub1 = cylinder(pos=vector(-6,0,60), axis=(2,0,0), radius=2)

az = 0
blen = 40
R = vector(-5.5,0,60)
X = vector(0,0,blen)
b1 = cylinder(pos=R, radius=1, axis=X.rotate(axis=(1,0,0),angle=(az+0)))
b2 = cylinder(pos=R, radius=1, axis=X.rotate(axis=(1,0,0),angle=(az+2*pi/3)))
b3 = cylinder(pos=R, radius=1, axis=X.rotate(axis=(1,0,0),angle=(az+4*pi/3)))

label(pos=(0,0,30), text='Flexible tower', xoffset=-20,color=color.black)
label(pos=R+X.rotate(axis=(1,0,0),angle=az)/2, text='Flexible blades',
      xoffset=20, color=color.black)

extrusion(pos=paths.arc(pos=(0,-10,0), radius=10, angle2=-2*pi/3),
          shape=shapes.circle(pos=(0,0), radius=0.5),
          color=color.red)
cone(pos=(10,-10,0),axis=(0,0,-3),radius=2, color=color.red)
cone(pos=(10*cos(2*pi/3),-10,10*sin(2*pi/3)),axis=(-3*sin(2*pi/3),0,3*cos(2*pi/3)),radius=2, color=color.red)

scene.center = (0,0,50)
scene.forward = (1,1,-0.5)
scene.up =(0,0,1)
