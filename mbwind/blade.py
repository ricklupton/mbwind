# -*- coding: utf-8 -*-
"""
Created on Tue May 08 14:13:23 2012

Represent a wind turbine blade, including mass and aerodynamic properties.

@author: Rick Lupton
"""

import numpy as np
from numpy import asarray, zeros_like
import re
from .modes import ModalRepresentation
from .old_io import convert_Bladed_attachment_modes
#from .elements import UniformBeam

def rotation_matrix_to_quaternions(R):
    q0 = 0.5 * np.sqrt(1 + R.trace())
    q1 =(R[2,1] - R[1,2]) / (4*q0)
    q2 =(R[0,2] - R[2,0]) / (4*q0)
    q3 =(R[1,0] - R[0,1]) / (4*q0)
    return np.array([q0,q1,q2,q3])

def system_shapes(system, x, q_dofs=None):
    """
    Find actual shapes and rotations of system
    """
    if q_dofs is not None:
        system.q.dofs[:] = q_dofs
    system.update_kinematics(0.0, False)

    beams = list(system.iter_elements())
    modes = np.zeros((len(beams)+1, 6))
    for ib,beam in enumerate(beams):
        qd = rotation_matrix_to_quaternions(beam.Rd)
        modes[1+ib,0:3] = beam.rd - [x[1+ib],0,0]
        modes[1+ib,3:6] = qd[1:]

    if False and np.abs(modes).max() > 0:
        modes /= np.abs(modes).max()
    return modes


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
            if blade_tower is not None:
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
        mstiff = BladedModule(prj, 'BSTIFFMB')

        # Geometry
        self.name = mgeom.bladename
        assert mgeom.nbe == len(mgeom.rj)
        self.radii = mgeom.rj         # station radius
        self.chord = mgeom.chord
        self.thickness = mgeom.bthick # percentage thickness
        self.twist = mgeom.twist

        # Mass
        self.density = mmass.mass

        # Stiffness
        self.EA = [0] * len(self.radii)
        self.EI_flap = mstiff.eiflap
        self.EI_edge = mstiff.eiedge

        # Remove repeated stations -- assume no split stations
        properties = ('radii', 'chord', 'thickness', 'twist', 'density',
                      'EA', 'EI_flap', 'EI_edge')
        for xx in properties:
            y = getattr(self, xx)
            assert y[::2] == y[1::2], "Cannot deal with split stations"
            setattr(self, xx, asarray(y[::2]))

        # Modes
        try:
            mmodes = BladedModule(prj, 'RMODE')
        except ModuleNotFoundError as e:
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
        rep = ModalRepresentation(self.radii,
                                  shapes=shapes,
                                  rotations=rotations,
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
        self.hubheight = mrcon.height

        mgeom = BladedModule(prj, 'TGEOM')
        self.model_type = int(mgeom.tmodel)

        if self.model_type == 2:
            # Axisymmetric
            self.nstations = mgeom.nte
            self.nelements = self.nstations - 1

            # Station positions in global coordinates (z up, x downwind)
            tj = np.array(mgeom.tj)
            self.stn_pos = np.c_[ np.zeros((self.nstations,2)), tj ]

            # Mass
            mtmass = BladedModule(prj, 'TMASS')
            towm = np.array(mtmass.towm)
            self.density = np.c_[ towm[:-1], towm[1:] ]
            self.polar_inertia = np.zeros_like(self.density) # axial tower has no polar inertia

            # Stiffness
            mtstiff = BladedModule(prj, 'TSTIFF')
            EI = mtstiff.eitow
            self.EIy = self.EIz = np.c_[ EI[:-1], EI[1:] ]
            self.EAx = np.nan * np.zeros_like(self.EIy)

        elif self.model_type == 3:
            # Multi member
            self.nstations = mgeom.nts
            self.nelements = mgeom.nte

            # Station positions in global coordinates (z up, x downwind)
            self.stn_pos = np.array(mgeom.tclocal).reshape((-1,3))

            # Element end stations
            self.element_stns = np.array(mgeom.elstns, dtype=int).reshape((-1,2)) - 1

            # Mass
            mtmass = BladedModule(prj, 'TMASS')
            self.density = np.array(mtmass.towm).reshape((-1,2))
            self.polar_inertia = np.array(mtmass.towpmi).reshape((-1,2))

            # Stiffness
            mtstiff = BladedModule(prj, 'TSTIFF')
            self.EIy = np.array(mtstiff.toweiy).reshape((-1, 2))
            self.EIz = np.array(mtstiff.toweiz).reshape((-1, 2))

        # Modes
        try:
            mmodes = BladedModule(prj, 'RMODE')
        except ModuleNotFoundError as e:
            print(e)
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

        # Mode names
        self.mode_names = []
        order = {0: 0, 1: 0, 2: 0, 3: 0}
        normal_names = [
            'Axial normal mode {}',
            'Transverse Y normal mode {}',
            'Transverse Z normal mode {}',
            'Torsional normal mode {}',
        ]
        attachment_names = [
            'Axial attachment mode',
            'Translational Y attachment mode',
            'Translational Z attachment mode',
            'Torsional attachment mode',
            'Rotation about Y attachment mode',
            'Rotation about Z attachment mode',
        ]
        for p,t in enumerate(self.mode_types):
            if t == 'N': # normal mode
                direction = np.argmax(np.max(np.abs(self.mode_data[:,0:4,p]),axis=0))
                order[direction] += 1
                name = normal_names[direction].format(order[direction])
            elif t == 'A': # attachment mode
                direction = np.argmax(np.abs(self.mode_data[-1,:,p]))
                name = attachment_names[direction]
            else:
                raise ValueError('Unknown mode type "%s"' % t)
            self.mode_names.append(name)

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

    def modal_rep(self, include_section_perp_inertia=True, rejig=False):
        """
        Return a ModalRepesentation containing the modal information
        """

        # XXX assume stations are defined in order from bottom to top
        x = self.stn_pos[:,2]
        x -= x[0] # start from bottom

        # Mass axis is given by other two coordinates of station positions
        mass_axis = self.stn_pos[:,:2]

        # Mode shapes and rotations
        shapes    = self.mode_data[:,:3,:]
        rotations = self.mode_data[:,3:,:]

        # Polar inertia
        # XXX smooth out discontinuities
        Ixx = np.r_[ self.polar_inertia[0,0],
                     0.5*(self.polar_inertia[1:,0] + self.polar_inertia[:-1,1]),
                     self.polar_inertia[-1,1]
        ]

        gyration_ratio = 1.0 if include_section_perp_inertia else None

        # Convert attachment modes
        mode_names, freqs, damping, shapes, rotations = \
            convert_Bladed_attachment_modes(self.mode_names, self.mode_freqs,
                                            self.mode_damping, shapes, rotations)

        rep = ModalRepresentation(x, shapes, rotations,
            density=self.density, freqs=freqs, damping=damping,
            mass_axis=mass_axis,
            section_inertia=Ixx,
            mode_names=mode_names,
            gyration_ratio=gyration_ratio,
            # rejig_attachment_modes=rejig,
            # EIy=self.EIy,
            # EIz=self.EIz,
        )
        return rep

#     def modal_analysis(self, Nmodes=None):
#         """
#         Calculate tower modes from finite element model.
#         """

#         print "Building model..."

#         # XXX assume stations are defined in order from bottom to top
#         x = self.stn_pos[:,2]
#         x -= x[0] # start from bottom
#         assert np.allclose(0, self.stn_pos[:,:2])

#         # Check all beams are uniform
#         assert np.allclose(0, np.diff(self.density))
#         assert np.allclose(0, np.diff(self.EIy))
#         assert np.allclose(0, np.diff(self.EIz))

#         # Build model
#         beams = [UniformBeam('beam%d'%(i+1), x[i+1]-x[i], self.density[i,0],
#                              0, self.EIy[i,0], self.EIz[i,0], 0, 0)
#                  for i in range(len(x)-1)]
#         for b1,b2 in zip(beams[:-1], beams[1:]):
#             b1.add_leaf(b2)
#         system = System(beams[0])

#         # Constrain extension and torsion
#         for b in beams:
#             system.prescribe(b, 0, 0, part=[0,3])

#         ## Fix distal end too...
#         #P_custom = np.zeros((2, system.lhs.shape[1]))
#         #b_custom = np.zeros(2)
#         #c_custom = np.zeros(2)
#         ## sum of y strains and z strains both equal zero
#         #P_custom[0,[b._istrain[1] for b in beams]] = 1
#         #P_custom[1,[b._istrain[2] for b in beams]] = 1
#         #system.custom_constraints = (P_custom, b_custom, c_custom)
#         #system.B = np.r_[ system.B[:-4], system.B[-2:] ]
#         #system.calc_projections()

#         print "Linearising..."

#         ## Linearise and find system modes
#         #linsys = LinearisedSystem(system)
#         #w, v = linsys.modes()
# #
#         #if Nmodes is None:
#             #Nmodes = len(w)
#         #assert Nmodes <= 6*len(beams)
#         #
#         #freqs = w[:Nmodes]
#         #damping = 0.01 * np.ones_like(freqs)
# #
#         ## Find actual shapes that correspond to beam strain mode shapes
#         #shapes    = np.zeros((len(beams)+1, 3, Nmodes))
#         #rotations = np.zeros((len(beams)+1, 3, Nmodes))
#         #for imode in range(Nmodes):
#             #m = system_shapes(system, x, v[:,imode]/50)
#             #shapes[:,:,imode] = m[:,0:3]
#             #rotations[:,:,imode] = m[:,3:6]
# #
#         #rep = ModalRepresentation(x, shapes, rotations,
#             #density=self.density,
#             #freqs  =freqs,
#             #damping=damping,
#         #)
#         rep = None

#         # Attachment modes: apply force to each distal direction in turn
#         att = np.zeros((len(beams)+1, 6, 4))
#         beams[-1].distal_load = lambda t: [0, 0.06, 0] #, 0, 0, 0]
#         #system.find_equilibrium()
#         att[:,:,0] = system_shapes(system, x)
#         #beams[-1].distal_load = lambda t: [0, 0, 1, 0, 0, 0]
#         #system.find_equilibrium()
#         #att[:,:,1] = system_shapes(system, x)
#         #beams[-1].distal_load = lambda t: [0, 0, 0, 0, 1, 0]
#         ##system.find_equilibrium()
#         #att[:,:,2] = system_shapes(system, x)
#         #beams[-1].distal_load = lambda t: [0, 0, 0, 0, 0, 1]
#         ##system.find_equilibrium()
#         #att[:,:,3] = system_shapes(system, x)

#         return rep, system, beams, att
