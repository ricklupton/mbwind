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
    def __init__(self, x, windfunc, aeroinfo):
        if isinstance(windfunc, np.ndarray):
            windt = windfunc[0]
            windv = windfunc[1:]
            windfunc = scipy.interpolate.interp1d(windt, windv)
            
        self.x = x # station positions
        self.windfunc = windfunc
        self.aeroinfo = aeroinfo
    
    def __call__(self, t, pos, ori, vel):
        """
        Calculate force distribution on blade
        
        Returns (N_x x 3) array of forces per length (in what coords?)
        """
        
        # XXX generalise
        Cd = 1.0
        diameter = 2.0

        force = zeros((len(self.x),3))
        
        # XXX should depend on position too
        global_windvel = self.windfunc(np.clip(t, self.windfunc.x[0],
                                               self.windfunc.x[-1]))
        
        # In Bladed, last station has zero loading
        for i in range(len(self.x)-1):
            r = pos[i]
            R = ori[i]
            v = vel[i]
            global_relvel = global_windvel - v
            local_relvel = dot(R.T, global_relvel)
            perp_relvel_sq = np.sum(local_relvel[1:]**2)
            perp_reldir = np.arctan2(local_relvel[2], local_relvel[1])
            
            dragforce = 0.5*1.225*Cd*diameter * perp_relvel_sq
            force[i,:] = dragforce * array([0, np.cos(perp_reldir), np.sin(perp_reldir)])
        
        return force