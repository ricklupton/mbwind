# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:24:13 2012

@author: Rick Lupton
"""

from __future__ import division
import time, sys
from collections import defaultdict
import numpy as np
from numpy import array, zeros, eye, dot, pi, r_, c_
import numpy.linalg as LA
import matplotlib.pylab as plt
from scipy.integrate import odeint
from scipy import sparse

import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})

import assemble


def rotmat_x(theta):
    return array([
        [1, 0,              0            ],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)],
    ])

def rotmat_y(theta):
    return array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)],
    ])

def rotmat_z(theta):
    return array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1],
    ])

def skewmat(vec):
    return array([
        [ 0,      -vec[2],  vec[1]],
        [ vec[2],  0,      -vec[0]],
        [-vec[1],  vec[0],  0     ],
    ])

def rotmatrix(xp):
    #pos = xp[:3]
    rot = xp[3:]
    return rot.reshape((3,3),order='F') # xp contains columns of rot. matrix

def combine_coords(rd, Rd):
    return np.r_[ rd, Rd.flatten(order='F') ]

def euler_param_mats(q):
    E = array([
        [-q[1],  q[0], -q[3],  q[2]],
        [-q[2],  q[3],  q[0], -q[1]],
        [-q[3], -q[2],  q[1],  q[0]],
    ])
    Ebar = array([
        [-q[1],  q[0],  q[3], -q[2]],
        [-q[2], -q[3],  q[0],  q[1]],
        [-q[3],  q[2], -q[1],  q[0]],
    ])
    return E, Ebar
 
# number of generalised position coordinates: 3 spatial plus 9 rotation matrix entries
NQ = 12
# number of generalised velocity coordinates: 3 spatial plus 3 angular velocity 
NQD = 6

class System(object):
    def __init__(self, first_element=None):
        self.mass = array([]) # mass matrix
        self.q = array([])
        self.qd = array([])
        self.first_element = first_element
        self.state_elements = []
        self.state_types = []
        self.states_by_type = defaultdict(list)
        
        # Prescribe ground node to be fixed by default
        self.prescribed_accels = {i: 0.0 for i in range(6)}
        
        self.iDOF = array([], dtype=bool)        
        
        if first_element is not None:
            self.init(first_element)
            
    def print_states(self):
        for el,type_ in zip(self.state_elements, self.state_types):
            print '{:<20}{:<20}'.format(el, type_)
    
    def iter_elements(self):
        yield self.first_element
        for el in self.first_element.iter_leaves():
            yield el
    
    def init(self, first_element):
        self.first_element = first_element
        
        # Reset state book-keeping
        self.state_elements = []
        self.state_types = []
        
        # Set up first node
        ground_ind = self.request_new_states(NQD, None, 'ground')
        
        # Set up first element to use first node as its proximal node, and set up
        # all its children (who will call request_new_states)
        self.first_element.setup_chain(self, ground_ind)
    
        # Now number of states is known, can size matrices and vectors
        Nq = len(self.state_elements)
        self.q    = zeros(Nq)
        self.qd   = zeros(Nq)
        self.qdd  = zeros(Nq)
        self.rhs  = zeros(Nq)
        #self.mass = sparse.coo_matrix((Nq,Nq))
        self.mass = zeros((Nq,Nq))
        self.iDOF = zeros(Nq, dtype=bool)
        self.iPrescribed = zeros(Nq, dtype=bool)
        self._update_indices()
        
    def _update_indices(self):
        self.iDOF[:] = False
        self.iPrescribed[:] = False
        
        # degrees of freedom are all strains...
        self.iDOF[self.states_by_type['strain']] = True
        
        # prescribed strains
        ipres = list(self.prescribed_accels.keys())
        self.iPrescribed[ipres] = True
    
    def request_new_states(self, num, elem, type):
        print 'element %s requesting %d new "%s" states' % (elem, num, type)
        n0 = len(self.state_elements)
        self.state_elements.extend([elem]*num)
        self.state_types.extend([type]*num)
        indices = range(n0, len(self.state_elements))
        self.states_by_type[type].extend(indices)
        return indices
    
    def update(self, dynamics=True):
        # Reset mass and constraint matrices if updating
        if dynamics:
            self.rhs[:]  = 0.0
            #self._mass_i = []
            #self._mass_j = []
            #self._mass_v = []
            self.mass[:] = 0.0

        # Update kinematics
        rp = zeros(3)
        Rp = eye(3)
        vp = zeros(NQD)
        self.first_element.update(rp, Rp, vp, dynamics)
        
        if dynamics:
            assemble.assemble(self.iter_elements(), self.mass, self.rhs)

        #if dynamics:
        #    Nq = len(self.state_elements)
        #    self.mass = sparse.coo_matrix( (self._mass_v, (self._mass_i,self._mass_j)),
        #                                   shape=(Nq,Nq) ).tocsr()
        
    def add_to_matrix(self, xind, yind, vals):
        #self._mass_i.extend(xind)
        #self._mass_j.extend(yind)
        #self._mass_v.extend(vals.tolist())
        for iv,xi in enumerate(xind):
            self.mass[xi,yind] += vals[iv,:]

    def add_to_rhs(self, ind, vals):
        self.rhs[ind] += vals
        
    def prescribe(self, ind, val):
        """
        Prescribe the accelerations listed in ind to equal val.
        They will be removed from the matrices when solving
        """
        for i in (np.iterable(ind) and ind or [ind]):
            if self.state_types[i] not in ('ground','strain'):
                raise Exception('Only ground node and strains can be prescribed')
            self.prescribed_accels[i] = val
        self._update_indices()
    
    def free(self, ind):
        """
        Remove a prescription
        """
        for i in (np.iterable(ind) and ind or [ind]):
            del self.prescribed_accels[i]
        self._update_indices()
        
    def solve(self):
        '''
        Solve for free accelerations, taking account of any prescribed accelerations
        '''
        
        self.qdd[:] = 0.0
        for i,a in self.prescribed_accels.items():
            self.qdd[i] = a

        prescribed_acc_forces = dot(self.mass, self.qdd)        
        if np.any(np.nonzero(prescribed_acc_forces)):
            print '* Prescribed acc forces:', np.nonzero(prescribed_acc_forces)
    
        # remove prescribed acceleration entries from mass matrix and RHS    
        # add the forces corresponding to prescribed accelerations back in
        M = self.mass[~self.iPrescribed,:][:,~self.iPrescribed]
        b = (self.rhs - prescribed_acc_forces)[~self.iPrescribed]
    
        # solve system for accelerations
        a = LA.solve(M, b)
        self.qdd[~self.iPrescribed] = a


        
gravity = 9.81

class Element(object):
    _ndistal = 0
    _nstrain = 0
    _nconstraints = 0
        
    def __init__(self, name):
        self.name = name
        self.children = []

        self.rp = zeros(3)
        self.Rp = eye(3)
        self.rd = np.tile(zeros(3), self._ndistal)
        self.Rd = np.tile(eye  (3), self._ndistal)
        self.xstrain = zeros(self._nstrain)

        self.vp = zeros(NQD)
        self.vd = zeros(NQD*self._ndistal)
        self.vstrain = zeros(self._nstrain)

        # Default [constant] parts of transformation matrices:
        # distal node velocities are equal to proximal, no strain effects
        # acceleration constraint -Fv2 = [ prox->dist, -I, strain->dist ] * [ v_prox, v_dist, v_strain ]
        self.F_vp = np.tile(eye(NQD), (self._ndistal,1))
        self.F_vd = -eye(NQD)
        self.F_ve = zeros((NQD*self._ndistal, self._nstrain))
        self.F_v2 = zeros( NQD*self._ndistal                )
        
        # Default [constant] parts of mass matrices
        nnodes = NQD * (1 + self._ndistal)
        self.mass_vv = zeros((nnodes,nnodes))
        self.mass_ve = zeros((nnodes,self._nstrain))
        self.mass_ee = zeros((self._nstrain,self._nstrain))
    
    def __str__(self):
        return self.name
    def __repr__(self):
        return '<%s "%s">' % (self.__class__.__name__, self.name)
        
    def add_leaf(self, elem, distnode=0):
        self.children.append((distnode,elem))
    
    def iter_leaves(self):
        for distnode,child in self.children:
            yield child
            for descendent in child.iter_leaves():
                yield descendent
    
    def plot_chain(self, ax):
        for linedata,opts in zip(self.shape(), self.shape_plot_options):
            ax.plot(linedata[:,0], linedata[:,1], linedata[:,2], **opts)
        for distnode,child in sorted(self.children):
            child.plot_chain(ax)
    
    def setup_chain(self, sys, prox_indices):
        assert len(prox_indices) == NQD # number of nodal coordinates
        self.system = sys
        self.iprox = prox_indices
        
        # Request new states for constraint multipliers
        self.imult = sys.request_new_states(self._nconstraints, self, 'constraint')

        # Request new states for internal strains
        self.istrain = sys.request_new_states(self._nstrain, self, 'strain')                
        
        # Request new states for distal node coordinates
        idist = []
        for i in range(self._ndistal):
            idist.append(sys.request_new_states(NQD, self, 'node'))        
        # Just keep a flat list of distal nodal indices
        self.idist = [i for sublist in idist for i in sublist]
        
        # Pass onto children for each distal node
        for distnode,child in sorted(self.children):
            child.setup_chain(sys, idist[distnode])

    def _inodes(self): return self.iprox + self.idist
    def _imass(self):  return self._inodes() + self.istrain
    
    def _check_ranges(self):
        pass

    def calc_external_forces(self):
        return zeros(NQD * (1 + self._ndistal))
        
    def calc_stresses(self):
        return zeros(self._nstrain)
    
    def update(self, rp, Rp, vp, matrices=True):
        # proximal values
        self.rp = rp
        self.Rp = Rp
        self.vp = vp
        
        # strain rates and accelerations
        self._check_ranges()
        self.xstrain = self.system.q[self.istrain]
        self.vstrain = self.system.qd[self.istrain]
            
        # Calculate kinematic transforms
        self.calc_kinematics()

        # find distal node values in terms of proximal node values
        self.rd,self.Rd = self.distal_pos()
        self.vd = dot(self.F_vp, self.vp     ) + \
                  dot(self.F_ve, self.vstrain)
        
        # Save back to system state vector
        #self.system.q[self.idist] = self.xd
        #self.system.qd[self.idist] = self.vd
        
        # Update mass and constraint matrices
        if matrices:
            # Calculate
            self.calc_mass()
            self._calc_applied_forces()
            self._calc_quadratic_forces()
            
            # Add to mass matrix
            #self.system.add_to_matrix(self._imass(), self._imass(), self._mass_matrix         )    
            #self.system.add_to_matrix(self.imult,    self._imass(), self._constraint_matrix   )
            #self.system.add_to_matrix(self._imass(), self.imult,    self._constraint_matrix.T )
            
            # Add to RHS
            #self.system.add_to_rhs(self._inodes(), self._applied_forces)
            #self.system.add_to_rhs(self.istrain,  -self._applied_stresses)
            #self.system.add_to_rhs(self._imass(),  self._quadratic_forces)
            #self.system.add_to_rhs(self.imult,     self.F_v2)
        
        # Pass onto children for each distal node
        for distnode,child in sorted(self.children):
            child.update(self.rd, self.Rd, self.vd, matrices)
    
    def _calc_applied_forces(self):
        # include gravity by default
        gravacc = array([0, 0, -gravity, 0, 0, 0,
                         0, 0, -gravity, 0, 0, 0,])
        self._applied_forces = dot(self.mass_vv, gravacc)
        self._applied_forces += self.calc_external_forces()
        self._applied_stresses = self.calc_stresses()
    
    def _calc_quadratic_forces(self):
        self._quadratic_forces = zeros(NQD*(1 + self._ndistal) + self._nstrain)
    
    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvv * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """
        pass

    def calc_mass(self):
        """
        Update mass matrices
        """
        pass

class Hinge(Element):
    _ndistal = 1
    _nstrain = 1
    _nconstraints = NQD
    
    def __init__(self, name, hinge_axis, post_transform=None):
        Element.__init__(self, name)
        self.hinge_axis = hinge_axis # proximal node coords
        self.stiffness = 0.0
        self.damping = 0.0
        if post_transform is None:
            post_transform = np.eye(3)
        self.post_transform = post_transform

    def _check_ranges(self):
        return
        xstrain = self.system.get_state(self.istrain, 0)
        if xstrain[0] > 2*pi:
            print 'Wrapping downwards'
            xstrain[0] -= 2*pi
            self.system.set_state(self.istrain, xstrain, 0)
        if xstrain[0] < 2*pi:
            print 'Wrapping upwards'
            xstrain[0] += 2*pi
            self.system.set_state(self.istrain, xstrain, 0)        

    
    def distal_pos(self):
        vs = skewmat(self.hinge_axis)
        th = self.xstrain[0]
        Rn = np.eye(3) + vs*np.sin(th) + 2*dot(vs,vs)*np.sin(th/2)**2
        Rc = self.post_transform
        return self.rp, dot(self.Rp, dot(Rn, Rc))

    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvv * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """
        n = dot(self.Rp, self.hinge_axis) # global hinge axis vector
        wps = skewmat(self.vp[3:])            # angular velocity matrix
        thd = self.vstrain[0]             # theta dot

        self.F_ve[3:,:] = n[:,np.newaxis]
        self.F_v2[3:] = thd*dot(wps, n)
    
    def shape(self):
        # proximal values
        return [
            array([self.rd]),
            self.rd + np.r_[ [dot(self.Rd,[1,0,0])], [[0,0,0]],
                             [dot(self.Rd,[0,1,0])], [[0,0,0]],
                             [dot(self.Rd,[0,0,1])] ]
        ]
    
    shape_plot_options = [
        {'marker': 's', 'ms': 4, 'c': 'r'},
        {'c': 'k', 'lw': 0.5}
    ]

    def calc_stresses(self):
        return array([self.stiffness * self.xstrain[0] + self.damping * self.vstrain[0]])

class RigidConnection(Element):
    _ndistal = 1
    _nstrain = 0
    _nconstraints = NQD
    
    def __init__(self, name, offset=None, rotation=None):
        Element.__init__(self, name)
        if offset is None:
            offset = zeros(3)
        if rotation is None:
            rotation = eye(3)            
        self.offset = offset
        self.rotation = rotation

    def distal_pos(self):
        rd = self.rp + dot(self.Rp, self.offset)
        Rd = dot(self.Rp, self.rotation)
        return rd, Rd
 
    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvv * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """

        wps = skewmat(self.vp[3:])
        xc = dot(self.Rp, self.offset)
        # Distal velocity rd_d = rd_p - xc \times w_p
        self.F_vp[0:3,3:6] = -skewmat(xc)
        # Distal velocity quadratic term w_p \times w_p \times xc
        self.F_v2[0:3] = dot( dot(wps,wps), xc )
  
    def shape(self):
        return [
            np.r_[ [self.rp], [self.rd] ],
            self.rd + np.r_[ [dot(self.Rd,[1,0,0])], [[0,0,0]],
                             [dot(self.Rd,[0,1,0])], [[0,0,0]],
                             [dot(self.Rd,[0,0,1])] ]
        ]
    
    shape_plot_options = [
        {'c': 'm', 'lw': 1},
        {'c': 'k', 'lw': 0.5}
    ]

# Slices to refer to parts of matrices
VP = slice(0,3)
WP = slice(3,6)
VD = slice(6,9)
WD = slice(9,12)

class EulerBeam(Element):
    _ndistal = 1
    _nstrain = 6
    _nconstraints = NQD
    
    def __init__(self, name, length, density, EA, EIy, EIz, GIx=0.0, Jx=0.0):
        '''
        Euler beam element.
        
         - length  : undeformed length of beam
         - density : mass per unit length
         - radius  : radius of beam, assumed circular for calculating stiffness
         - EA      : Extensional stiffness
         - GIx     : torsional stiffness (G*Ix*kx)
         - EIy     : bending stiffness about y axis
         - EIz     : bending stiffness about z axis
         
        '''
        Element.__init__(self, name)        
        self.length = length
        self.linear_density = density
        self.damping = 0.1
        self.EA = EA
        self.GIx = GIx
        self.EIy = EIy
        self.EIz = EIz
        self.Jx = Jx
        self._initial_calcs()
        
    def _initial_calcs(self):
        m = self.linear_density # mass per unit length
        l0 = self.length
        # Integrals of interpolating factors
        # with t2 = l0*(xi - p1 + p2)
        self._mass_coeffs = array([
            [13*l0*m/35, 11*l0**2*m/210, 9*l0*m/70, 13*l0**2*m/420],
            [0, l0**3*m/105, 13*l0**2*m/420, l0**3*m/140],
            [0, 0, 13*l0*m/35, 11*l0**2*m/210],
            [0, 0, 0, l0**3*m/105]
        ])
        
        # with t2 = l0*(xi - p1 - p2)
        #self._mass_coeffs = array([
        #    [13*l0*m/35, -l0**2*m/105, 9*l0*m/70, 13*l0**2*m/420],
        #    [0, 2*l0**3*m/105, -31*l0**2*m/420, -l0**3*m/84],
        #    [0, 0, 13*l0*m/35, 11*l0**2*m/210],
        #    [0, 0, 0, l0**3*m/105]
        #])
        
        # Set constant parts of mass matrix
        c = self._mass_coeffs
        I = eye(3)
        self.mass_vv[VP,VP] = c[0,0]*I
        self.mass_vv[VP,VD] = c[0,2]*I
        self.mass_vv[VD,VP] = c[0,2]*I
        self.mass_vv[VD,VD] = c[2,2]*I

    def _beam_rfuncs(self, s):
        # centreline in local coords in terms of generalised strains
        l = self.length
        return array([
            [s, 0,               0,               0, 0,             0,           ],
            [0, 3*s**2 - 2*s**3, 0,               0, 0,            -l*(s**2-s**3)],
            [0, 0,               3*s**2 - 2*s**3, 0, l*(s**2-s**3), 0            ],
        ])
    
    def _beam_qfuncs(self, s):
        # rotation in local coords in terms of generalised strains
        l = self.length
        return 0.5 * array([
            [0, 0,            0,            s, 0,          0          ],
            [0, 0,           -6*(s-s**2)/l, 0, 3*s**2-2*s, 0          ],
            [0, 6*(s-s**2)/l, 0,            0, 0,          3*s**2-2*s ],
        ])

    def _beam_centreline(self, s, xstrain):
        # return centreline in proximal frame
        S = self._beam_rfuncs(s)
        X0 = array([self.length*s, 0, 0])
        return X0 + dot(S, xstrain)
    
    def _beam_centreline_vel(self, s, vstrain):
        # return centreline in proximal frame
        S = self._beam_rfuncs(s)
        return dot(S, vstrain)
    
    def _beam_rotation(self, s, xstrain):
        # return cross-section rotation (Euler parameters in proximal frame)
        B = self._beam_qfuncs(s)
        q1,q2,q3 = dot(B, xstrain).flatten()
        if np.amax(np.abs([q1,q2,q3])) > 1:
            raise Exception('Euler parameters excced 1')
        q0 = np.sqrt(1.0 - q1**2 - q2**2 - q3**2)
        return array([q0,q1,q2,q3])

    def _beam_rotation_vel(self, s, xstrain, vstrain):
        # return cross-section rotation (Euler parameters in proximal frame)
        B = self._beam_qfuncs(s)
        qd1,qd2,qd3 = dot(B, vstrain).flatten()
        q0,q1,q2,q3 = self._beam_rotation(1.0, xstrain)
        qd0 = -(q1*qd1 + q2*qd2 + q3*qd3)/q0
        return array([qd0,qd1,qd2,qd3])
        
    def _beam_R(self, s, xstrain):
        # rotation matrix of cross-section in proximal frame
        q = self._beam_rotation(s, xstrain)
        E,Ebar = euler_param_mats(q)
        return dot(E,Ebar.T)
    
    def _beam_wrel(self, s, xstrain, vstrain):
        q0, q1, q2, q3  = self._beam_rotation(s, xstrain)
        qd0,qd1,qd2,qd3 = self._beam_rotation_vel(s, xstrain, vstrain)
        E,Ebar = euler_param_mats((q0,q1,q2,q3))
        return 2*dot(E,[qd0,qd1,qd2,qd3])
    
    def distal_pos(self):
        X  = array([self.length,0,0]) + self.xstrain[0:3]
        R  = self._beam_R(1.0, self.xstrain)
        rd = self.rp + dot(self.Rp,X)
        Rd = dot(self.Rp, R)
        return rd, Rd
    
    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvv * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """
        X  = array([self.length,0,0]) + self.xstrain[0:3] # Distal pos in local coords
        Xd = self.vstrain[0:3]
        wps = skewmat(self.vp[3:])

        Sr = self._beam_rfuncs(1.0)
        Sq = self._beam_qfuncs(1.0)
        
        q  = self._beam_rotation(1.0, self.xstrain)
        qd = self._beam_rotation_vel(1.0, self.xstrain, self.vstrain)
        wrel = self._beam_wrel(1.0, self.xstrain, self.vstrain)
        E, Ebar  = euler_param_mats(q)
        Ed,Edbar = euler_param_mats(qd)
        assert np.allclose(q [1:]*2, self.xstrain[3:])
        assert np.allclose(qd[1:]*2, self.vstrain[3:])

        # distal vel
        assert np.allclose(skewmat(dot(self.Rp,X)), dot(self.Rp,dot(skewmat(X),self.Rp.T)))
        self.F_vp[0:3,3:6] = -dot(self.Rp,dot(skewmat(X),self.Rp.T))

        # Distal vel due to strain
        self.F_ve[0:3,:] = dot(self.Rp, Sr)
        # Distal ang vel due to strain
        # TODO: is this right?
        self.F_ve[3:6,:] = dot(self.Rp, dot(2*E[:,1:], Sq))

        # Quadratic velocity terms
        self.F_v2[0:3] = 2*dot(wps,     dot(self.Rp,Xd)) + dot(dot(wps,wps), dot(self.Rp, X   ))
        self.F_v2[3:6] = 2*dot(self.Rp, dot(Ed,qd     )) + dot(wps,          dot(self.Rp, wrel))

    def shape(self, N=20):
        # proximal values
        shape = [dot(self.Rp, self._beam_centreline(i/N, self.xstrain)) for i in range(N+1)]
        return [
            self.rp + array(shape),
            self.rd + np.r_[ [dot(self.Rd,[1,0,0])], [[0,0,0]],
                             [dot(self.Rd,[0,1,0])], [[0,0,0]],
                             [dot(self.Rd,[0,0,1])] ],
            np.r_[ [self.rp], [self.rp + dot(self.Rp,[self.length,0,0])], [self.rd] ]
        ]
        
    shape_plot_options = [
        {'c': 'g', 'marker': 'o', 'lw': 2, 'ms': 1},
        {'c': 'k', 'lw': 0.5},
        {'c': 'k', 'lw': 2, 'alpha': 0.3},
    ]
    
    def _calc_quadratic_forces(self):
        # dist prox strain
        ep = self.Rp[:,0] # unit vectors along elastic line
        ed = self.Rd[:,0]       
        eps = skewmat(ep)
        eds = skewmat(ed)
        wps = skewmat(self.vp[3:]) # angular velocity matrices
        wds = skewmat(self.vd[3:])
        c = self._mass_coeffs # shorthand
        self._quadratic_forces = np.r_[
            c[0,1]*dot(        dot(wps,wps) ,ep) + c[0,3]*dot(        dot(wds,wds) ,ed),
            c[1,1]*dot(dot(eps,dot(wps,wps)),ep) + c[1,3]*dot(dot(eps,dot(wds,wds)),ed),
            c[1,2]*dot(        dot(wps,wps) ,ep) + c[2,3]*dot(        dot(wds,wds) ,ed),
            c[1,3]*dot(dot(eds,dot(wps,wps)),ep) + c[3,3]*dot(dot(eds,dot(wds,wds)),ed),
            zeros(self._nstrain) # quadratic stresses
        ]
    
    def calc_mass(self):
        nps = skewmat(self.Rp[:,0]) # unit vectors along elastic line
        nds = skewmat(self.Rd[:,0])        
        # lumped inertias xx
        Jbar = zeros((3,3)); Jbar[0,0] = self.linear_density * self.Jx
        Jxxp = dot(self.Rp, dot(Jbar, self.Rp.T))
        Jxxd = dot(self.Rd, dot(Jbar, self.Rd.T))

        c = self._mass_coeffs # shorthand
        m = self.mass_vv
        
        #[VP,VP] constant
        m[VP,WP] = -c[0,1]*nps
        #[VP,VD] constant
        m[VP,WD] = -c[0,3]*nds
        
        m[WP,VP] =  m[VP,WP].T
        m[WP,WP] = -c[1,1]*dot(nps,nps) + Jxxp
        m[WP,VD] =  c[1,2]*nps
        m[WP,WD] = -c[1,3]*dot(nps,nds)
        
        #[VD,VP] constant
        m[VD,WP] =  m[WP,VD].T
        #[VD,VD] constant
        m[VD,WD] = -c[2,3]*nds
        
        m[WD,VP] =  m[VP,WD].T
        m[WD,WP] =  m[WP,WD].T
        m[WD,VD] =  m[VD,WD].T
        m[WD,WD] = -c[3,3]*dot(nds,nds) + Jxxd
        
    #def calc_external_forces(self):
    #    f = zeros(12)
    #    #f[7] = 10
    #    return f
        
    def calc_stresses(self):
        EA = self.EA
        GIx,EIy,EIz = self.GIx, self.EIy, self.EIz
        l = self.length
        stiffness = array([
            [EA, 0,           0,           0,     0,          0         ],
            [0,  12*EIz/l**3, 0,           0,     0,         -6*EIz/l**2],
            [0,  0,           12*EIy/l**3, 0,     6*EIy/l**2, 0,        ],
            [0,  0,           0,           GIx/l, 0,          0         ],
            [0,  0,           6*EIy/l**2,  0,     4*EIy/l,    0         ],
            [0, -6*EIz/l**2,  0,           0,     0,          4*EIz/l   ],
        ])
        return dot(stiffness, self.xstrain) + dot(self.damping, dot(stiffness, self.vstrain))
        
################################################

def rk4(derivs, y0, t):
    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros( (len(t),), np.float_)
    else:
        yout = np.zeros( (len(t), Ny), np.float_)
    yout[0] = y0
    try:
        for i in np.arange(len(t)-1):
            thist = t[i]
            dt = t[i+1] - thist
            dt2 = dt/2.0
            y0 = yout[i]
            k1 = np.asarray(derivs(y0, thist))
            k2 = np.asarray(derivs(y0 + dt2*k1, thist+dt2))
            k3 = np.asarray(derivs(y0 + dt2*k2, thist+dt2))
            k4 = np.asarray(derivs(y0 + dt*k3, thist+dt))
            yout[i+1] = y0 + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)
    except IndexError as e:
        print
        print '! Error evaluating function gradients:'
        print '!', e
        print '!'
        raise
    return yout

def solve_system(system, t):
    def func(y, ti):
        # update system state with states and state rates
        # y constains [strains, d(strains)/dt]
        system.q[system.iDOF]  = y[:len(y)/2]
        system.qd[system.iDOF] = y[len(y)/2:]
        system.update()
        
        # solve system
        system.solve()
                
        # new state vector is [strains_dot, strains_dotdot]
        return np.r_[ system.qd[system.iDOF], system.qdd[system.iDOF] ]


    y0 = np.r_[ system.q[system.iDOF], system.qd[system.iDOF] ]
    
    print 'Running simulation...',
    sys.stdout.flush()
    tstart = time.clock()

    y = odeint(func, y0, t)
    #y = rk4(func, y0, t)

    print 'done'
    print '%.1f seconds elapsed' % (time.clock() - tstart)
    
    return y

