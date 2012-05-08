# -*- coding: utf-8 -*-
"""
Sources of loading: aerodynamic and hydrodynamic

Created on Fri Apr 13 12:33:51 2012

@author: Rick Lupton
"""

import numpy as np
from numpy import array, dot, zeros, zeros_like
import scipy.interpolate

class BladeLoading(object):
    def __init__(self, blade, windfunc, aeroinfo):
        if isinstance(windfunc, np.ndarray):
            windt = windfunc[0]
            windv = windfunc[1:]
            windfunc = scipy.interpolate.interp1d(windt, windv)
            
        self.blade = blade
        self.windfunc = windfunc
        self.aeroinfo = aeroinfo
    
    def __call__(self, t, pos, ori, vel):
        """
        Calculate force distribution on blade
        
        Returns (N_x x 3) array of forces per length (in what coords?)
        """
        
        # XXX generalise
        Cd = 0.5

        force = zeros((len(self.blade.radii),3))
        
        # XXX should depend on position too
        self.global_windvel = self.windfunc(np.clip(t, self.windfunc.x[0],
                                               self.windfunc.x[-1]))
        
        self.global_relvel = self.global_windvel[None,:] - vel
        self.local_relvel = np.einsum('hji,hj->hi', ori, self.global_relvel)
        self.relspeed = np.sqrt(np.sum(self.local_relvel[:,1:]**2, axis=1))
        self.reldir = zeros_like(self.relspeed)
        self.tipvel = vel[-1]
        
        # In Bladed, last station has zero loading
        for i in range(len(self.blade.radii)-1):
            diameter = self.blade.chord[i]
            self.reldir[i] = np.arctan2(self.local_relvel[i,2], self.local_relvel[i,1])
            
            dragforce = 0.5*1.225*Cd*diameter * self.relspeed[i]**2
            force[i,:] = dragforce * array([0, np.cos(self.reldir[i]), np.sin(self.reldir[i])])
        
        return force