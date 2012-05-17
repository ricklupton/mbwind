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
        
        if hasattr(self, '_parser_%s'%name.lower()):
            items = getattr(self, '_parser_%s'%name.lower())(self.text)
        else:
            items = self._default_parser(self.text)
        
        for key,value in items.items():
            setattr(self, key.lower(), value)
    
    def _iter_records(self, text):
        # Split on newlines that are not followed by space, to allow multilines
        for record in re.split(r'\n(?![ \t])', text):
            if not record: continue
            # Split first word by any whitespace
            record = [xx.strip() for xx in record.split(None, 1)] 
            key, value = (record[0], '') if (len(record) == 1) else record
            
            # Process numeric and list values
            splitvalue = splitquoted(value)
            if len(splitvalue) > 1:
                value = [xx.strip() for xx in splitvalue]
                try: value = [float(xx) for xx in value]
                except ValueError: pass
            else:
                try: value = float(value)
                except ValueError: pass
            yield key, value
    
    def _parser_rmode(self, text):
        '''
        Parse mode shapes: separate tower and blade modes
        '''
        result = {}
        blade_tower = None
        for key,value in self._iter_records(text):
            if key.lower() == 'nblade':
                blade_tower = 'blade_'
            elif key.lower() == 'ntower':
                blade_tower = 'tower_'
            result[blade_tower + key.lower()] = value
        return result
        
    def _default_parser(self, text):
        result = {}
        for key,value in self._iter_records(text):
            result[key.lower()] = value
        return result

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
            self.mode_data = np.zeros((len(self.radii), 6, 0))
        else:
            self.nmodes = int(mmodes.blade_nblade)
            self.mode_types = mmodes.blade_type
            self.mode_freqs = asarray(mmodes.blade_freq)
            self.mode_damping = asarray(mmodes.blade_damp)
            if hasattr(mmodes, 'crypt'):
                raise NotImplementedError("Cannot read encrypted modes")
                
            self.mode_data = np.zeros((len(self.radii), 6, self.nmodes))
            # Rearrange axes: Bladed uses z along blade, I'm using x along blade
            #   Bladed x -> -z (my)
            #          y ->  y
            #          z ->  x
            for i in range(self.nmodes):
                for j0,j1 in enumerate((2,1,0,5,4,3)):
                    self.mode_data[:,j1,i] = getattr(mmodes, 'blade_md%02d%d'%(1+i,1+j0))                    
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

class Tower(object):
    """
    Hold all the properties of a tower, and read from a Bladed file
    """
    def __init__(self, filename):
        with open(filename, 'r') as f:
            prj = f.read()
        
        # Geometry
        mrcon = BladedModule(prj, 'RCON')
        mgeom = BladedModule(prj, 'TGEOM')
        self.hubheight = mrcon.height
        self.nstations = mgeom.nts
        self.nelements = mgeom.nte
        
        # Station positions in global coordinates (z up, x downwind)
        self.stn_pos = np.array(mgeom.tclocal).reshape((-1,3))
        
        # Element end stations
        self.element_stns = np.array(mgeom.elstns, dtype=int).reshape((-1,2)) - 1
        
        # Mass
        mtmass = BladedModule(prj, 'TMASS')
        self.density = np.array(mtmass.towm).reshape((-1,2))
        
        # Modes
        try:
            mmodes = BladedModule(prj, 'RMODE')
        except ModuleNotFoundError as e:
            print e
            self.nmodes = 0
            self.mode_types = []
            self.mode_freqs = []
            self.mode_damping = []
            self.mode_data = np.zeros((self.nstations, 6, 0))
        else:
            self.nmodes = int(mmodes.tower_ntower)
            self.mode_types = mmodes.tower_type
            self.mode_freqs = asarray(mmodes.tower_freq)
            self.mode_damping = asarray(mmodes.tower_damp)
            if hasattr(mmodes, 'crypt'):
                raise NotImplementedError("Cannot read encrypted modes")
                
            self.mode_data = np.zeros((self.nstations, 6, self.nmodes))
            # Rearrange axes: Bladed uses z along blade, I'm using x along blade
            #   Bladed x -> -z (my)
            #          y ->  y
            #          z ->  x
            for i in range(self.nmodes):
                for j0,j1 in enumerate((2,1,0,5,4,3)):
                    self.mode_data[:,j1,i] = getattr(mmodes, 'tower_md%02d%d'%(1+i,1+j0))                    
                self.mode_data[:,2,i] *= -1
                self.mode_data[:,5,i] *= -1

    @property
    def element_lengths(self):
        end1 = self.stn_pos[self.element_stns[:,0]]
        end2 = self.stn_pos[self.element_stns[:,1]]
        lengths = np.sum((end2-end1)**2, axis=-1)**0.5
        return lengths
    
    @property
    def total_mass(self):
        # element masses using trapezium rule
        masses = np.sum(self.density, axis=-1) * self.element_lengths / 2
        return np.sum(masses)
    
    @property
    def total_inertia(self):
        # XXX should calc inertia about base
        return np.zeros((3,3))
    
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