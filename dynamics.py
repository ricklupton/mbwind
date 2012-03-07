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
        xp = combine_coords(zeros(3), eye(3))
        vp = zeros(NQD)
        self.first_element.update(xp, vp, dynamics)
        
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
        self.xp = combine_coords(zeros(3), eye(3))
        self.xd = np.tile(combine_coords(zeros(3), eye(3)), self._ndistal)
        self.xstrain = zeros(self._nstrain)
        self.vp = zeros(NQD)
        self.vd = zeros(NQD*self._ndistal)
        self.vstrain = zeros(self._nstrain)
    
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

    def calc_external_forces(self, xp, xd, xstrain, vp, vd, vstrain):
        return zeros(NQD * (1 + self._ndistal))
        
    def calc_stresses(self, xp, xd, xstrain, vp, vd, vstrain):
        return zeros(self._nstrain)
    
    def update(self, xp, vp, matrices=True):
        # proximal values
        self.xp = xp
        self.vp = vp
        #xp = self.system.get_state(self.iprox, 0)
        #vp = self.system.get_state(self.iprox, 1)
        
        # strain rates and accelerations
        self._check_ranges()
        self.xstrain = self.system.q[self.istrain]
        self.vstrain = self.system.qd[self.istrain]
            
        # find distal node values in terms of proximal node values
        self.xd = self.distal_pos(self.xp, self.xstrain)
        self.vd = dot(self.F_vv(self.xp, self.xd, self.xstrain), self.vp     ) + \
                  dot(self.F_ve(self.xp, self.xd, self.xstrain), self.vstrain)
        
        # XXX
        self._F_v2 = self.F_v2(self.xp, self.xd, self.xstrain, self.vp, self.vd, self.vstrain)
                
        # Save back to system state vector
        self.system.q[self.idist] = self.xd
        self.system.qd[self.idist] = self.vd
        
        # Update mass and constraint matrices
        if matrices:
            # Calculate
            self._calc_mass_matrix(self.xp, self.xd, self.xstrain)
            self._calc_constraint_matrix(self.xp, self.xd, self.xstrain)
            self._calc_applied_forces(self.xp, self.xd, self.xstrain, self.vp, self.vd, self.vstrain)
            self._calc_quadratic_forces(self.xp, self.xd, self.xstrain, self.vp, self.vd, self.vstrain)
            
            # Add to mass matrix
            self.system.add_to_matrix(self._imass(), self._imass(), self._mass_matrix         )    
            self.system.add_to_matrix(self.imult,    self._imass(), self._constraint_matrix   )
            self.system.add_to_matrix(self._imass(), self.imult,    self._constraint_matrix.T )
            
            # Add to RHS
            self.system.add_to_rhs(self._inodes(), self._applied_forces)
            self.system.add_to_rhs(self.istrain,  -self._applied_stresses)
            self.system.add_to_rhs(self._imass(),  self._quadratic_forces)
            self.system.add_to_rhs(self.imult,     self._F_v2)
        
        # Pass onto children for each distal node
        for distnode,child in sorted(self.children):
            child.update(self.xd, self.vd, matrices)
    
    def _calc_constraint_matrix(self, xp, xd, xstrain):
        # acceleration constraint 0 = [ -prox->dist, I, -strain->dist ] * [ v_prox, v_dist, v_strain ]
        self._constraint_matrix = np.c_[ 
            -self.F_vv(xp, xd, xstrain), np.eye(NQD), -self.F_ve(xp, xd, xstrain)
        ]

    def _calc_mass_matrix(self, xp, xd, xstrain):
        m_vv = self.mass_matrix_vv(xp, xd, xstrain)
        m_ve = self.mass_matrix_ve(xp, xd, xstrain)
        m_ee = self.mass_matrix_ee(xp, xd, xstrain)
        self._mass_matrix = np.r_[ np.c_[m_vv,   m_ve],
                                   np.c_[m_ve.T, m_ee] ]

    def mass_matrix_vv(self, xp, xd, xstrain):
        n = NQD * (1 + self._ndistal)
        return zeros((n,n))
    def mass_matrix_ve(self, xp, xd, xstrain):
        n = NQD * (1 + self._ndistal)
        return zeros((n,self._nstrain))
    def mass_matrix_ee(self, xp, xd, xstrain):
        return zeros((self._nstrain,self._nstrain))
    
    def _calc_applied_forces(self, xp, xd, xstrain, vp, vd, vstrain):
        # include gravity by default
        n = NQD * (1 + self._ndistal)
        gravacc = array([0, 0, -gravity, 0, 0, 0,
                         0, 0, -gravity, 0, 0, 0,])
        self._applied_forces = dot(self._mass_matrix[:n,:n], gravacc)
        self._applied_forces += self.calc_external_forces(xp, xd, xstrain, vp, vd, vstrain)
        self._applied_stresses = self.calc_stresses(xp, xd, xstrain, vp, vd, vstrain)
    
    def _calc_quadratic_forces(self, xp, xd, strain, vp, vd, vstrain):
        self._quadratic_forces = zeros(NQD*(1 + self._ndistal) + self._nstrain)
    
    def F_vv(self, xp, xd, xstrain):
        # acceleration transfer func
        return zeros((NQD*self._ndistal,NQD))    
    
    def F_ve(self, xp, xd, xstrain):
        # strain constraint
        return zeros((NQD*self._ndistal, self._nstrain))
    
    def F_v2(self, xp, xd, xstrain, vp, vd, vstrain):
        return zeros(NQD*self._ndistal)

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

    
    def distal_pos(self, xp, xstrain):
        Rp = rotmatrix(xp)
        vs = skewmat(self.hinge_axis)
        th = xstrain[0]
        Rn = np.eye(3) + vs*np.sin(th) + 2*dot(vs,vs)*np.sin(th/2)**2
        Rc = self.post_transform
        
        rd = xp[:3]
        Rd = dot(Rp, dot(Rn, Rc))
        return combine_coords(rd, Rd)
    
    def F_vv(self, xp, xd, xstrain):
        # acceleration transfer func
        return np.eye(NQD)
    
    def F_ve(self, xp, xd, xstrain):
        # strain constraint
        Rp = rotmatrix(xp)
        n = dot(Rp, self.hinge_axis) # global hinge axis vector
        return np.r_[ zeros(3), n ].reshape((-1,1))
    
    def F_v2(self, xp, xd, xstrain, vp, vd, vstrain):
        Rp = rotmatrix(xp)
        n = dot(Rp, self.hinge_axis) # global hinge axis vector
        wps = skewmat(vp[3:])
        thd = vstrain[0]
        return np.r_[ [0, 0, 0], thd*dot(wps, n) ].T
    
    def shape(self):
        # proximal values
        rd = self.xd[:3] #self.system.get_state(self.iprox, 0)
        Rd = rotmatrix(self.xd)
        return [
            array([rd]),
            rd + np.r_[ [dot(Rd,[1,0,0])], [[0,0,0]], [dot(Rd,[0,1,0])], [[0,0,0]], [dot(Rd,[0,0,1])] ]
        ]
    
    shape_plot_options = [
        {'marker': 's', 'ms': 4, 'c': 'r'},
        {'c': 'k', 'lw': 0.5}
    ]

    def calc_stresses(self, xp, xd, xstrain, vp, vd, vstrain):
        return array([self.stiffness * xstrain[0] + self.damping * self.vstrain[0]])

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
        
    def distal_pos(self, xp, xstrain):
        Rp = rotmatrix(xp)
        rd = xp[:3] + dot(Rp, self.offset)
        Rd = dot(Rp, self.rotation)
        return combine_coords(rd, Rd)
    
    def F_vv(self, xp, xd, xstrain):
        Rp = rotmatrix(xp)
        xcs = skewmat(dot(Rp, self.offset))
        return np.c_[
            np.r_[ eye(3),      -xcs    ],
            np.r_[ zeros((3,3)), eye(3) ],
        ]
        
    def F_v2(self, xp, xd, xstrain, vp, vd, vstrain):
        Rp = rotmatrix(xp)
        xc = dot(Rp, self.offset)
        wps = skewmat(vp[3:])
        return np.r_[ dot(dot(wps,wps), xc), zeros(3) ]
    
    def shape(self):
        rp = self.xp[:3]
        rd = self.xd[:3]
        Rd = rotmatrix(self.xd)
        return [
            np.r_[ [rp], [rd] ],
            rd + np.r_[ [dot(Rd,[1,0,0])], [[0,0,0]], [dot(Rd,[0,1,0])], [[0,0,0]], [dot(Rd,[0,0,1])] ]
        ]
    
    shape_plot_options = [
        {'c': 'm', 'lw': 1},
        {'c': 'k', 'lw': 0.5}
    ]

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
        #self._mass_coeffs = array([
        #    [13*l0*m/35, 11*l0**2*m/210, 9*l0*m/70, 13*l0**2*m/420],
        #    [0, l0**3*m/105, 13*l0**2*m/420, l0**3*m/140],
        #    [0, 0, 13*l0*m/35, 11*l0**2*m/210],
        #    [0, 0, 0, l0**3*m/105]
        #])
        
        # with t2 = l0*(xi - p1 - p2)
        self._mass_coeffs = array([
            [13*l0*m/35, -l0**2*m/105, 9*l0*m/70, 13*l0**2*m/420],
            [0, 2*l0**3*m/105, -31*l0**2*m/420, -l0**3*m/84],
            [0, 0, 13*l0*m/35, 11*l0**2*m/210],
            [0, 0, 0, l0**3*m/105]
        ])

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
    
    def distal_pos(self, xp, xstrain):
        Rp = rotmatrix(xp)
        rp = xp[:3] 
        X  = self._beam_centreline(1.0, xstrain)
        R  = self._beam_R(1.0, xstrain)
        rd = rp + dot(Rp,X)
        Rd = dot(Rp, R)

        assert np.allclose(X, [self.length+xstrain[0],xstrain[1],xstrain[2]])
        q = self._beam_rotation(1.0, xstrain)
        assert np.allclose(q[1:]*2, xstrain[3:])
        
        return combine_coords(rd, Rd)
    
    def F_vv(self, xp, xd, xstrain):
        # acceleration transfer func
        Rp = rotmatrix(xp)
        X  = self._beam_centreline(1.0, xstrain) # local vector P->D
        return np.c_[
            np.r_[ np.eye(3),      -dot(Rp,dot(skewmat(X),Rp.T)) ],
            np.r_[ np.zeros((3,3)), np.eye(3)                    ]
        ]
    
    def F_ve(self, xp, xd, xstrain):
        Rp = rotmatrix(xp)
        Sr = self._beam_rfuncs(1.0)
        Sq = self._beam_qfuncs(1.0)
        
        q = self._beam_rotation(1.0, xstrain)
        E,Ebar = euler_param_mats(q)
        assert np.allclose(q[1:]*2, xstrain[3:])

        return np.r_[ dot(Rp, Sr), dot(Rp, dot(2*E[:,1:], Sq)) ]
    
    def F_v2(self, xp, xd, xstrain, vp, vd, vstrain):
        Rp = rotmatrix(xp)
        X  = self._beam_centreline(1.0, xstrain) # local vector P->D
        Xd = self._beam_centreline_vel(1.0, vstrain)
        wps = skewmat(vp[3:])
        
        qd = self._beam_rotation_vel(1.0, xstrain, vstrain)
        assert np.allclose(qd[1:]*2, vstrain[3:])
        Ed,Edbar = euler_param_mats(qd)
        wrel = self._beam_wrel(1.0, xstrain, vstrain)
        return np.r_[
            2*dot(wps, dot(Rp,Xd)) + dot(dot(wps,wps), dot(Rp,X)),
            2*dot(Rp, dot(Ed,qd))  + dot(wps, dot(Rp, wrel))
        ]
        
    def shape(self, N=20):
        # proximal values
        rp = self.xp[:3]
        rd = self.xd[:3]
        Rp = rotmatrix(self.xp)
        Rd = rotmatrix(self.xd)
        shape = [dot(Rp, self._beam_centreline(i/N, self.xstrain)) for i in range(N+1)]
        return [
            rp + array(shape),
            rd + np.r_[ [dot(Rd,[1,0,0])], [[0,0,0]], [dot(Rd,[0,1,0])], [[0,0,0]], [dot(Rd,[0,0,1])] ],
            np.r_[ [rp], [rp + dot(Rp,[self.length,0,0])], [rd] ]
        ]
        
    shape_plot_options = [
        {'c': 'g', 'marker': 'o', 'lw': 2, 'ms': 1},
        {'c': 'k', 'lw': 0.5},
        {'c': 'k', 'lw': 2, 'alpha': 0.3},
    ]
    
    def _calc_quadratic_forces(self, xp, xd, strain, vp, vd, vstrain):
        # dist prox strain
        Rp = rotmatrix(xp)
        Rd = rotmatrix(xd)
        ep = Rp[:,0] # unit vectors along elastic line
        ed = Rd[:,0]       
        eps = skewmat(ep)
        eds = skewmat(ed)
        wps = skewmat(vp[3:]) # angular velocity matrices
        wds = skewmat(vd[3:])
        c = self._mass_coeffs # shorthand
        self._quadratic_forces = np.r_[
            c[0,1]*dot(        dot(wps,wps) ,ep) + c[0,3]*dot(        dot(wds,wds) ,ed),
            c[1,1]*dot(dot(eps,dot(wps,wps)),ep) + c[1,3]*dot(dot(eps,dot(wds,wds)),ed),
            c[1,2]*dot(        dot(wps,wps) ,ep) + c[2,3]*dot(        dot(wds,wds) ,ed),
            c[1,3]*dot(dot(eds,dot(wps,wps)),ep) + c[3,3]*dot(dot(eds,dot(wds,wds)),ed),
            zeros(self._nstrain) # quadratic stresses
        ]
    
    def mass_matrix_vv(self, xp, xd, xstrain):
        Rp = rotmatrix(xp)
        Rd = rotmatrix(xd)
        nps = skewmat(Rp[:,0]) # unit vectors along elastic line
        nds = skewmat(Rd[:,0])        
        c = self._mass_coeffs # shorthand
        I = eye(3)
        # lumped inertias xx
        Jbar = zeros((3,3)); Jbar[0,0] = self.linear_density * self.Jx
        Jxxp = dot(Rp, dot(Jbar, Rp.T))
        Jxxd = dot(Rd, dot(Jbar, Rd.T))
        mvv = np.r_[
            np.c_[ c[0,0]*I,   -c[0,1]*nps,                 c[0,2]*I,    -c[0,3]*nds                 ],
            np.c_[ c[0,1]*nps, -c[1,1]*dot(nps,nps) + Jxxp, c[1,2]*nps,  -c[1,3]*dot(nps,nds)        ],
            np.c_[ c[0,2]*I,   -c[1,2]*nps,                 c[2,2]*I,    -c[2,3]*nds                 ],
            np.c_[ c[0,3]*nds, -c[1,3]*dot(nds,nps),        c[2,3]*nds,  -c[3,3]*dot(nds,nds) + Jxxd ],
        ]
        return mvv
        
    def calc_external_forces(self, xp, xd, xstrain, vp, vd, vstrain):
        f = zeros(12)
        #f[7] = 10
        return f
        
    def calc_stresses(self, xp, xd, xstrain, vp, vd, vstrain):
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
        return dot(stiffness, xstrain) + dot(self.damping, dot(stiffness, vstrain))
        
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

