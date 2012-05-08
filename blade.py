# -*- coding: utf-8 -*-
"""
Created on Tue May 08 14:13:23 2012

Represent a wind turbine blade, including mass and aerodynamic properties.

@author: Rick Lupton
"""

import numpy as np
from numpy import asarray
import re
from linearisation import ModalRepresentation

def splitquoted(string):
    """Split on spaces, keeping quoted things together"""
    parts = re.findall(r"('[^']*'|[^\n ,]+)", string)
    return [xx.replace("'", "") for xx in parts]

class ModuleNotFoundError(Exception):
    pass

class BladedModule(object):
    def __init__(self, prj, name):
        self._module_name = name
        
        pattern = r"""
            ^[ \t]*MSTART[ \t]+{modname}\b.*\n  # MSTART as 1st word then modname
            ((?:(?![ \t]*MEND).*\n)*)           # then lines not starting MEND
        """.format(modname=name)
        module = re.search(pattern, prj, re.X+re.M+re.I)
        if not module:
            raise ModuleNotFoundError("Could not find module '{}'".format(name))
        self.text = module.group(1)
        
        items = re.findall(r"^[ \t]*([^ \t]+)[ \t]+(.*)", self.text, re.M)
        for key,value in items:
            splitvalue = splitquoted(value)
            if len(splitvalue) > 1:
                value = [xx.strip() for xx in splitvalue]
                try: value = [float(xx) for xx in value]
                except ValueError: pass
            else:
                try: value = float(value)
                except ValueError: pass
            setattr(self, key.lower(), value)

class Blade(object):
    """
    Hold all the properties of a blade, and read from a Bladed file
    """
    def __init__(self, filename):
        with open(filename, 'r') as f:
            prj = f.read()
        mgeom = BladedModule(prj, 'BGEOMMB')
        mmass = BladedModule(prj, 'BMASSMB')
        
        
        # Geometry
        self.name = mgeom.bladename        
        assert mgeom.nbe == len(mgeom.rj)
        self.radii = mgeom.rj         # station radius
        self.chord = mgeom.chord
        self.thickness = mgeom.bthick # percentage thickness
        self.twist = mgeom.twist
                
        # Mass
        self.density = mmass.mass
        
        # Remove repeated stations -- assume no split stations
        for xx in ('radii', 'chord', 'thickness', 'twist', 'density'):
            y = getattr(self, xx)
            assert y[::2] == y[1::2], "Cannot deal with split stations"
            setattr(self, xx, asarray(y[::2]))
        
        # Modes
        try:
            mmodes = BladedModule(prj, 'RMODE')
        except ModuleNotFoundError as e:
            print e
            self.nmodes = 0
            self.mode_types = []
            self.mode_freqs = []
            self.mode_damping = []
            self.mode_data = np.zeros((6, 0, len(self.radii)))
        else:
            self.nmodes = int(mmodes.nblade)
            self.mode_types = mmodes.type
            self.mode_freqs = asarray(mmodes.freq)
            self.mode_damping = asarray(mmodes.damp)
            if hasattr(mmodes, 'crypt'):
                raise NotImplementedError("Cannot read encrypted modes")
                
            self.mode_data = np.zeros((len(self.radii), 6, self.nmodes))
            # Rearrange axes: Bladed uses z along blade, I'm using x along blade
            #   Bladed x -> -z (my)
            #          y ->  y
            #          z ->  x
            for i in range(self.nmodes):
                for j0,j1 in enumerate((2,1,0,5,4,3)):
                    self.mode_data[:,j1,i] = getattr(mmodes, 'md%02d%d'%(1+i,1+j0))                    
                self.mode_data[:,2,i] *= -1
                self.mode_data[:,5,i] *= -1

    
    def modal_rep(self):
        """
        Return a ModalRepesentation containing the modal information
        """
        
        shapes    = self.mode_data[:,:3,:]
        rotations = self.mode_data[:,3:,:]
        #kgyr = np.zeros_like(bmf.radii)
        #mass_axis = bmf.mass_axis
        rep = ModalRepresentation(self.radii, shapes, rotations,
            density=self.density,
            freqs  =self.mode_freqs,
            damping=self.mode_damping,
#            section_inertia=kgyr,
        )
        return rep