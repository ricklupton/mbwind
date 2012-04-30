# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:24:13 2012

@author: Rick Lupton
"""

from __future__ import division
import time, sys
import operator
from collections import defaultdict
import numpy as np
from numpy import array, zeros, eye, dot, pi, cos, sin
import numpy.linalg as LA
import scipy.linalg
import scipy.optimize
#import matplotlib.pylab as plt
from scipy.integrate import ode, simps


import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]})

import assemble

eps_ijk = zeros((3,3,3))
eps_ijk[0,1,2] = eps_ijk[1,2,0] = eps_ijk[2,0,1] =  1
eps_ijk[2,1,0] = eps_ijk[1,0,2] = eps_ijk[0,2,1] = -1


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

def qrot3(q):
    q1,q2,q3 = q
    q0 = np.sqrt(1.0 - q1**2 - q2**2 - q3**2)
    assert not np.isnan(q0)
    return array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)    ],
        [2*(q1*q2 + q0*q3),     1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)    ],
        [2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1),     1 - 2*(q1**2 + q2**2)],
    ])

def euler_param_E(q):
    return array([
        [-q[1],  q[0], -q[3],  q[2]],
        [-q[2],  q[3],  q[0], -q[1]],
        [-q[3], -q[2],  q[1],  q[0]],
    ])

# number of generalised position coordinates: 3 spatial plus 9 rotation matrix entries
NQ = 12
# number of generalised velocity coordinates: 3 spatial plus 3 angular velocity
NQD = 6

#class StateArray(object):
#    def __init__(self):
#        self.reset()
#    
#    def __getitem__(self, index):
#        if self._array is None:
#            raise RuntimeError("Need to call allocate() first")
#        if index in self.index_map:
#            print "Remapping {} to {}".format(index, self.index_map[index])
#            index = self.index_map[index]
#        return self._array[index]
#    
#    def __setitem__(self, index, value):
#        if self._array is None:
#            raise RuntimeError("Need to call allocate() first")
#        if index in self.index_map:
#            print "Remapping {} to {}".format(index, self.index_map[index])
#            index = self.index_map[index]
#        self._array[index] = value
#    
#    def allocate(self):
#        if self._array not is None:
#            raise RuntimeError("Already allocated")
#        N = len(self.elements)
#        self._array = np.array(N)
#    
#    def reset(self):
#        self.owners = []
#        self.types = []
#        self.index_map = {}
#        self._array = None
#    
#    def new_states(self, owner, type_, num):
#        if self._array is None:
#            raise RuntimeError("Already allocated, call reset()")
#        self.owners.extend( [owner]*num )
#        self.types.extend ( [type_]*num )


class ArrayProxy(object):
    """Delegate getitem and setitem to a reduced part of the target array"""
    def __init__(self, target, subset=None):
        if subset is None:
            subset = range(len(target))
        self.subset = subset
        self._target = target
    
    def __getitem__(self, index):
        return self._target[self.subset[index]]
    
    def __setitem__(self, index, value):
        self._target[self.subset[index]] = value

class StateArray(object):
    def __init__(self):
        self.reset()
    
    def indices(self, index):
        if isinstance(index, basestring):
            index = self.named_indices[index]
        if isinstance(index, int):
            index = slice(index, index+1)
        return index
        
    def __getitem__(self, index):
        if self._array is None:
            raise RuntimeError("Need to call allocate() first")
        return self._array[self.indices(index)]
    
    def __setitem__(self, index, value):
        if self._array is None:
            raise RuntimeError("Need to call allocate() first")
        self._array[self.indices(index)] = value
    
    def __len__(self):
        return len(self.owners)
    
    def allocate(self):
        if self._array is not None:
            raise RuntimeError("Already allocated")
        self._array = np.zeros(len(self))
        self.dofs = ArrayProxy(self._array, [])
    
    def reset(self):
        self.owners = []
        self.types = []
        self.named_indices = {}
        self.dofs = ArrayProxy([])
        self._array = None
        
    def new_states(self, owner, type_, num, name=None):
        if self._array is not None:
            raise RuntimeError("Already allocated, call reset()")
        self.owners.extend( [owner]*num )
        self.types.extend ( [type_]*num )
        if name is not None:
            if name in self.named_indices:
                raise KeyError("{} already exists".format(name))
            N = len(self.owners)
            self.named_indices[name] = slice(N-num, N)
        
    def indices_by_type(self, types):
        "Return all state indices of the given type"
        if not isinstance(types, (list,tuple)):
            types = (types,)
        return [i for i,t in enumerate(self.types) if t in types]
    
    def names_by_type(self, types):
        "Return all named states of the given type"
        if not isinstance(types, (list,tuple)):
            types = (types,)
        return [name for name,ind in sorted(self.named_indices.iteritems(), key=operator.itemgetter(1))
                if self.get_type(name) in types]
    
    def by_type(self, types):
        "Return all elements of the given type"
        index = self.indices_by_type(types)
        return self._array[index]
    
    def get_type(self, index):
        types = self.types[self.indices(index)]
        if not all([t == types[0] for t in types]):
            raise ValueError("index refers to different types")
        if types:
            return types[0]
        return None

class System(object):
    def __init__(self, first_element=None):
        # State vectors
        self.q = StateArray()
        self.qd = StateArray()
        self.qdd = StateArray()
        # Joint reaction forces (ON prox node)
        self.joint_reactions = StateArray()
        
        # System LHS matrix and RHS vector
        self.lhs = array([]) 
        self.rhs = array([])
        
        # Bookkeeping
        self.first_element = first_element
        self.time = 0.0

        # Prescribe ground node to be fixed by default
        # velocity and acceleration
        # actually (-b,-c), where
        #   b = -\phi_t, i.e. partial derivative of constraint wrt time
        #   c = -\dot{\phi_t} - \dot{phi_q}\dot{q}
        # assuming these constraints relate only to a single DOF, then the
        # second term in c is zero, and c is just the time derivative of b.
        # Either scalars of a callable can be supplied. The callable will be
        # called with the current time at each step.
        self.prescribed_dofs = {i: (0.0,0.0) for i in range(6)}

        if first_element is not None:
            self.init(first_element)

    def print_states(self, q=False):
        if q: states = self.q
        else: states = self.qd
        print '     Element             Type                Prescribed'
        print '-------------------------------------------------------'
        for i,(el,type_) in enumerate(zip(states.owners, states.types)):
            prescribed = (i in self.prescribed_dofs) and '*' or ' '
            print '{:<5}{:<20}{:<20} {}'.format(i,el, type_, prescribed)

    def iter_elements(self):
        yield self.first_element
        for el in self.first_element.iter_leaves():
            yield el

    def print_info(self):
        print 'System:\n'
        for element in self.iter_elements():
            element.print_info()
            print

    def init(self, first_element):
        self.first_element = first_element

        # Reset state vectors
        for states in (self.q, self.qd, self.qdd, self.joint_reactions):
            states.reset()
        self._node_counter = 0
        
        # Set up first node
        ground_ind = self._new_states(None, 'ground', NQD, NQ, 'ground')

        # Set up first element to use first node as its proximal node, and set up
        # all its children (who will call request_new_states)
        self.first_element.setup_chain(self, ground_ind)

        # Now number of states is known, can size matrices and vectors
        for states in (self.q, self.qd, self.qdd, self.joint_reactions):
            states.allocate()
        N = len(self.qd)
        self.rhs  = zeros(N)
        self.lhs = zeros((N,N))
        
        self.iPrescribed = zeros(N, dtype=bool)
        self._update_indices()
        
        # Give elements a chance to finish setting up (make views into global vectors)
        for element in self.iter_elements():
            element.finish_setup()

    ##################################################################
    ###   Book-keeping   #############################################
    ##################################################################
    def _new_states(self, elem, type_, num, num_q, name):
        # Nodes have different number of entries in q vector than qd and qdd
        self.q  .new_states(elem, type_, num_q, name)
        self.qd .new_states(elem, type_, num  , name)
        self.qdd.new_states(elem, type_, num  , name)
        self.joint_reactions.new_states(elem, type_, num, name)
        return name
    
    def request_new_node(self, elem):
        name = 'node-{}'.format(self._node_counter)
        self._node_counter += 1
        return self._new_states(elem, 'node', NQD, NQ, name)
    
    def request_new_strains(self, elem, num):
        name = elem.name + '-strains'
        return self._new_states(elem, 'strain', num, num, name)
    
    def request_new_constraints(self, elem, num):
        # make identifier for these states
        name = elem.name + '-constraints'
        return self._new_states(elem, 'constraint', num, num, name)

    def _update_indices(self):
        # Keep track of which are the free DOFs in state arrays
        for states in (self.q, self.qd, self.qdd, self.joint_reactions):
            states.dofs.subset = [i for i in states.indices_by_type('strain')
                                  if i not in self.prescribed_dofs]
        
        self.iPrescribed[:] = False

        # prescribed strains
        ipres = list(self.prescribed_dofs.keys())
        self.iPrescribed[ipres] = True

        # real coords, node and strains
        ir = self.qd.indices_by_type(('ground', 'node', 'strain'))
        self.iReal = zeros(len(self.qd), dtype=bool)
        self.iReal[ir] = True

        # update B matrix
        indep_coords = self.qd.dofs.subset
        B = zeros((len(indep_coords),len(self.qd)), dtype=bool)
        for iz,iq in enumerate(indep_coords):
            B[iz,iq] = True
        self.B = B[:,self.iReal]

    def get_constraints(self):
        """
        Return constraint jacobian \Phi_q and the vectors b and c
        """
        # Constraints relating nodes defined by elements
        # They don't vary partially with time, so b=0. c contains derivatives
        # of constraint equations themselves.
        P_nodal = self.lhs[np.ix_(~self.iReal,self.iReal)] # rows:constraints, columns:coords
        c_nodal = self.rhs[~self.iReal]
        b_nodal = np.zeros_like(c_nodal)
        
        # Add extra user-specified constraints for prescribed accelerations etc
        # These are assumed to relate always to just one DOF.
        ipres_orig    = np.nonzero(self.iPrescribed)[0]
        ipres_reduced = np.nonzero(self.iPrescribed[self.iReal])[0]
        P_prescribed = np.zeros((len(ipres_orig), P_nodal.shape[1]))
        c_prescribed = np.zeros(len(ipres_orig))
        b_prescribed = np.zeros(len(ipres_orig))
        for i in range(len(ipres_orig)):
            b,c = self.prescribed_dofs[ipres_orig[i]]
            if callable(c): c = c(self.time)
            if callable(b): b = b(self.time)
            if c is None or b is None:
                raise Exception("Cannot calculate constraints if a prescribed"\
                    " DOF is None")
            P_prescribed[i,ipres_reduced[i]] = 1
            c_prescribed[i] = c
            b_prescribed[i] = b
        
        return (np.r_[P_nodal, P_prescribed],
                np.r_[b_nodal, b_prescribed],
                np.r_[c_nodal, c_prescribed])

    def calc_projections(self):
        """
        Calculate the projection matrices S and R which map from indep. to all coords
        
        qd  = R zd  +  S b
        qdd = R zdd +  S c
        """
        f,n = self.B.shape
        P,b,c = self.get_constraints()
        SR = LA.inv(np.r_[P, self.B])        
        self.S = SR[:,:n-f]
        self.R = SR[:,n-f:]
        self.Sc = dot(self.S,c)
        self.Sb = dot(self.S,b)

    def update_kinematics(self, time=None, calculate_matrices=True):
        if time is not None:
            self.time = time

        # Update prescribed strains
        for i,(b,c) in self.prescribed_dofs.items():
            if callable(b): b = b(self.time)
            if callable(c): c = c(self.time)
            if b is not None:
                self.qd[i] = b
            else:
                pass # leave it for the integration
            if c is not None:
                self.qdd[i] = c
            else:
                self.qdd[i] = 0.0

        # Reset mass and constraint matrices if updating
        if calculate_matrices:
            self.lhs[:] = 0.0
            self.rhs[:] = 0.0            

        # Update kinematics
        r0 = self.q[:3]
        R0 = self.q[3:12].reshape((3,3))
        r0[:] = 0.0
        R0[:,:] = eye(3)
        for element in self.iter_elements():
            element.update(calculate_matrices)
        
        # Assemble mass matrices, constraint matrices and RHS vectors
        if calculate_matrices:
            assemble.assemble(self.iter_elements(), self.lhs, self.rhs)

    def prescribe(self, ind, part=None, vel=None, acc=None):
        """
        Prescribe the DOFs listed in ind with the velocity and acceleration
        constraints given.
        
        vel = -b = \phi_t, i.e. partial derivative of constraint wrt time
        acc = -c = \dot{\phi_t} + \dot{phi_q}\dot{q}
        
        The specified DOFs will be removed from the matrices when solving.
        """
        ind_slice = self.qd.indices(ind)
        ind_range = range(ind_slice.start, ind_slice.stop)
        if part is not None:
            ind_range = ind_range[part]
        for i in ind_range:
            if self.qd.types[i] not in ('ground','strain'):
                raise Exception('Only ground node and strains can be prescribed')
            self.prescribed_dofs[i] = (vel, acc)
        self._update_indices()
        
        # XXX messy
        ind_slice = self.q.indices(ind)
        ind_range = range(ind_slice.start, ind_slice.stop)
        if part is not None:
            ind_range = ind_range[part]
        for i in ind_range:
            self.q.dofs.subset.remove(i)

    def free(self, ind):
        """
        Remove a prescription
        """
        for i in self.qd.indices(ind):
            del self.prescribed_dofs[i]
        self._update_indices()

    def solve_accelerations(self):
        '''
        Solve for free accelerations, taking account of any prescribed accelerations
        '''

        #self.calc_projections()
        #self.qdd[:] = 0.0
        #for i,(vel,acc) in self.precribed_dofs.items():
        #    self.qdd[i] = a
        #self.qdd[self.iReal] = self.Sc
        #print self.Sc

        prescribed_acc_forces = dot(self.lhs, self.qdd._array)
        #if np.any(np.nonzero(prescribed_acc_forces)):
        #    print '* Prescribed acc forces:', np.nonzero(prescribed_acc_forces)

        # remove prescribed acceleration entries from mass matrix and RHS
        # add the forces corresponding to prescribed accelerations back in
        M = self.lhs[~self.iPrescribed,:][:,~self.iPrescribed]
        b = (self.rhs - prescribed_acc_forces)[~self.iPrescribed]

        # solve system for accelerations
        a = LA.solve(M, b)
        
        new_qdd = self.qdd._array.copy()
        new_qdd[~self.iPrescribed] = a
        return new_qdd
    
    def solve_reactions(self):
        """
        Iterate backwards down tree solving for joint reaction forces.
        Assumes the motion has been solved, i.e. used q, qd, and qdd
        """
        self.joint_reactions[:] = 0.0
        for elem in reversed(list(self.iter_elements())):
            elem.iter_reactions()

    def find_equilibrium(self):
        """
        Solve static equalibrium problem, using currently set initial velocities
        and accelerations.
        """
        def func(z):
            # Update system and matrices
            self.q.dofs[:] = z
            self.update()
            self.calc_projections()
            # eval residue: external forces, quadratic terms and internal forces...
            Q = self.rhs[self.iReal]
            # ... and inertial loads
            qdd = self.Sc # prescribed accelerations
            ma = dot(self.mass[np.ix_(self.iReal,self.iReal)], qdd)
            return dot(self.R.T, (Q - ma))
        
        q0 = scipy.optimize.fsolve(func, self.q.dofs[:])
        self.q.dofs[:] = q0

    def eval_residue(self, z, zd, zdd, parts=False):
        self.q.dofs[:] = z
        self.qd.dofs[:] = zd
        self.update_kinematics()
        self.calc_projections()

        RtM = dot(self.R.T, self.lhs[np.ix_(self.iReal,self.iReal)])
        Mr = dot(RtM,self.R)
        Qr = dot(self.R.T,self.rhs[self.iReal])
        
        if parts:
            return {
                'M':   dot(Mr, zdd),
                'Q':  -Qr,
                'Sc':  dot(RtM, self.Sc),
            }
        else:
            return dot(Mr, zdd) - Qr + dot(RtM, self.Sc)


class ReducedSystem(object):
    def __init__(self, full_system):
        full_system.calc_projections()
        full_M = full_system.lhs[np.ix_(full_system.iReal,full_system.iReal)]
        full_Q = full_system.rhs[full_system.iReal]
        R = full_system.R
        
        self.M = dot(R.T, dot(full_M, R))
        self.Q = dot(R.T, (full_Q - dot(full_M, full_system.Sc)))

gravity = 9.81

class Element(object):
    _ndistal = 0
    _nstrain = 0
    _nconstraints = 0

    def __init__(self, name):
        self.name = name
        self.children = [[]] * self._ndistal

        self.rp = zeros(3)
        self.Rp = eye(3)
        self.rd = np.tile(zeros(3), self._ndistal)
        self.Rd = np.tile(eye  (3), self._ndistal)
        self.xstrain = zeros(self._nstrain)

        self.vp = zeros(NQD)
        self.vd = zeros(NQD*self._ndistal)
        self.vstrain = zeros(self._nstrain)
        
        self.ap = zeros(NQD)
        self.ad = zeros(NQD*self._ndistal)
        self.astrain = zeros(self._nstrain)

        # Default [constant] parts of transformation matrices:
        # distal node velocities are equal to proximal, no strain effects
        # acceleration constraint -Fv2 = [ prox->dist, -I, strain->dist ] * [ a_prox, a_dist, a_strain ]
        self.F_vp = np.tile(eye(NQD), (self._ndistal,1))
        self.F_vd = -eye(NQD)
        self.F_ve = zeros((NQD*self._ndistal, self._nstrain))
        self.F_v2 = zeros( NQD*self._ndistal                )

        # Default [constant] parts of mass matrices
        nnodes = NQD * (1 + self._ndistal)
        self.mass_vv = zeros((nnodes,nnodes))
        self.mass_ve = zeros((nnodes,self._nstrain))
        self.mass_ee = zeros((self._nstrain,self._nstrain))
        self.quad_forces = zeros(NQD*(1 + self._ndistal))
        self.quad_stress = zeros(self._nstrain)

        # External forces and stresses
        self.applied_forces = zeros(NQD * (1 + self._ndistal))
        self.applied_stress = zeros(self._nstrain)

    def __str__(self):
        return self.name
    def __repr__(self):
        return '<%s "%s">' % (self.__class__.__name__, self.name)

    def _ind(self, i):
        ind = self.system.qd.indices(i)
        return range(ind.start, ind.stop)
    def _iprox(self):
        return self._ind(self.iprox)
    def _idist(self):
        if self.idist:
            return self._ind(self.idist[0])
        return []
    def _istrain(self):
        return self._ind(self.istrain)
    def _imult(self):
        return self._ind(self.imult)

    def add_leaf(self, elem, distnode=0):
        self.children[distnode].append(elem)

    def iter_leaves(self):
        for node_children in self.children:
            for child in node_children:
                yield child
                for descendent in child.iter_leaves():
                    yield descendent

    def plot_chain(self, ax):
        for linedata,opts in zip(self.shape(), self.shape_plot_options):
            ax.plot(linedata[:,0], linedata[:,1], linedata[:,2], **opts)
        for node_children in self.children:
            for child in node_children:
                child.plot_chain(ax)
        
    def print_info(self):
        print '{!r}:'.format(self)
        print '    prox node: {}'.format(self.iprox)
        if self._nstrain:
            print '    strains: {} ({})'.format(self._nstrain, self.istrain)
        if self._nconstraints:
            print '    constraints: {} ({})'.format(self._nconstraints, self.imult)
        if self._ndistal:
            print '    distal nodes: {}'.format(', '.join(self.idist))

    def setup_chain(self, system, prox_indices):
        self.system = system
        self.iprox = prox_indices

        # Request new states for constraint multipliers
        self.imult = system.request_new_constraints(self, self._nconstraints)

        # Request new states for internal strains
        self.istrain = system.request_new_strains(self, self._nstrain)

        # Request new states for distal node coordinates
        self.idist = [system.request_new_node(self) for i in range(self._ndistal)]
        
        # Pass onto children for each distal node
        for inode,node_children in zip(self.idist,self.children):
            for child in node_children:
                child.setup_chain(system, inode)
        
    def finish_setup(self):
        system = self.system
        
        # Setup views into global arrays
        self.rp = system.q[self.iprox][:3]
        self.Rp = system.q[self.iprox][3:].reshape((3,3))
        self.vp = system.qd[self.iprox]
        self.ap = system.qdd[self.iprox]
        self.xstrain = system.q[self.istrain]
        self.vstrain = system.qd[self.istrain]
        self.astrain = system.qdd[self.istrain]
        
        assert len(self.idist) <= 1
        for idist in self.idist:
            self.rd = system.q[idist][:3]
            self.Rd = system.q[idist][3:].reshape((3,3))
            self.vd = system.qd[idist]
            self.ad = system.qdd[idist]

        

    def _check_ranges(self):
        pass

    def calc_external_loading(self):
        self._set_gravity_force()

    def update(self, calculate_matrices=True):
        
        # check all strains are within range (e.g. wrap round)
        self._check_ranges()
        
        # Calculate kinematic transforms
        self.calc_kinematics()

        # find distal node values in terms of proximal node values
        if self._ndistal > 0:
            self.calc_distal_pos()
            self.vd[:] = dot(self.F_vp, self.vp     ) + \
                         dot(self.F_ve, self.vstrain)
            self.ad[:] = dot(self.F_vp, self.ap     ) + \
                         dot(self.F_ve, self.astrain) + self.F_v2

        # Update mass and constraint matrices
        if calculate_matrices:
            # Calculate
            self.calc_mass()
            self.calc_external_loading()
    
    def iter_reactions(self):
        """
        Calculate reaction forces on proximal node, based on current nodal
        accelerations, and distal reaction forces
        """
        # inertial and applied forces acting ON all element nodes
        a_nodal = np.r_[ self.ap, self.ad ]
        elem = -dot(self.mass_vv, a_nodal) + -dot(self.mass_ve, self.astrain) +\
                       -self.quad_forces + self.applied_forces
         
        
        
        # calculate forces acting ON proximal nodes:
        #   Fprox + elem_forces_on_prox + sum of distal_forces = 0
        Fprox = -elem[0:3].copy()
        Mprox = -elem[3:6].copy()
        for i,dist in enumerate(self.idist):
            # forces acting ON distal nodes
            distal_forces = elem[6+6*i:12+6*i] - self.system.joint_reactions[dist]
            relcross = skewmat(self.rd[3*i:3*i+3] - self.rp)
            Fprox += -distal_forces[:3]
            Mprox += -distal_forces[3:]-dot(relcross,distal_forces[:3])
        
        #if self.name == 'blade':
        #    print self.station_velocities()[-1] #, self.system.qdd[self.idist]
        
        self.system.joint_reactions[self.system.qd.named_indices[self.iprox]] \
            += np.r_[ Fprox, Mprox ]

    def _set_gravity_force(self):
        # include gravity by default
        gravacc = np.tile([0, 0, -gravity, 0, 0, 0], 1 + self._ndistal)
        self.applied_forces = dot(self.mass_vv, gravacc)

    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvv * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """
        pass

    def calc_mass(self):
        """
        Update mass matrices and quadratic force vector
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
        if self.xstrain[0] > 2*pi:
            print 'Wrapping downwards'
            self.xstrain[0] -= 2*pi
        if self.xstrain[0] < 2*pi:
            print 'Wrapping upwards'
            self.xstrain[0] += 2*pi

    def calc_distal_pos(self):
        vs = skewmat(self.hinge_axis)
        th = self.xstrain[0]
        Rn = np.eye(3) + vs*np.sin(th) + 2*dot(vs,vs)*np.sin(th/2)**2
        Rc = self.post_transform
        self.rd[:] = self.rp
        self.Rd[:,:] = dot(self.Rp, dot(Rn, Rc))

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

    def calc_external_loading(self):
        self.applied_stress[0] = self.stiffness*self.xstrain[0] + self.damping*self.vstrain[0]

class PrismaticJoint(Element):
    _ndistal = 1
    _nstrain = 1
    _nconstraints = NQD

    def __init__(self, name, axis, post_transform=None):
        Element.__init__(self, name)
        self.axis = axis # proximal node coords
        self.stiffness = 0.0
        self.damping = 0.0
        if post_transform is None:
            post_transform = np.eye(3)
        self.post_transform = post_transform

    def calc_distal_pos(self):
        n = dot(self.Rp, self.axis) # axis in global frame
        self.rd[:] = self.rp + self.xstrain[0]*n
        self.Rd[:,:] = dot(self.Rp, self.post_transform)

    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvv * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """
        n = dot(self.Rp, self.axis) # global axis vector
        self.F_ve[:3,:] = n[:,np.newaxis]

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

    def calc_external_loading(self):
        self.applied_stress[0] = self.stiffness*self.xstrain[0] + self.damping*self.vstrain[0]


class FreeJoint(Element):
    _ndistal = 1
    _nstrain = 6
    _nconstraints = NQD

    def __init__(self, name, stiffness=None, damping=None, post_transform=None):
        Element.__init__(self, name)
        if stiffness is None: stiffness = zeros((6,6))
        if damping is None: damping = zeros((6,6))
        self.stiffness = stiffness
        self.damping = damping
        self.post_transform = post_transform

        # Constant parts of transformation matrices
        self.F_ve = eye(6)

    def calc_distal_pos(self):
        Rd = dot( dot(rotmat_z(self.xstrain[3]),
                      rotmat_y(self.xstrain[4])),
                      rotmat_x(self.xstrain[5]))
        if self.post_transform is not None:
            Rd = dot(Rd, self.post_transform)
        self.rd[:] = self.rp + self.xstrain[0:3]
        self.Rd[:,:] = Rd

    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvv * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """
        # Angle strains to angular velocity depend on current angles
        F = self.F_ve[3:6,3:6]
        a1,a2,a3 = self.xstrain[3:6]
        b1,b2,b3 = self.vstrain[3:6]
        F[0:3,0] = [cos(a1)*cos(a2), sin(a3)*cos(a2), -sin(a2)]
        F[0:2,1] = [-sin(a3), cos(a3)]
        self.F_v2[3:6] = [
            -b1*b2*cos(a3)*sin(a2) - b1*b3*sin(a3)*cos(a2) - b2*b3*cos(a3),
            -b1*b2*sin(a3)*sin(a2) - b1*b3*cos(a3)*cos(a2) - b2*b3*sin(a3),
            -b1*b2*cos(a2),
        ]

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

    def calc_external_loading(self):
        self.applied_stress[:] = dot(self.stiffness,self.xstrain) + \
                                 dot(self.damping,  self.vstrain)

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

    def calc_distal_pos(self):
        self.rd[:] = self.rp + dot(self.Rp, self.offset)
        self.Rd[:,:] = dot(self.Rp, self.rotation)

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

class RigidBody(Element):
    _ndistal = 0
    _nstrain = 0
    _nconstraints = 0

    def __init__(self, name, mass, inertia=None, Xc=None):
        '''
        Rigid body element with only one node.
        Defined by mass and inertia tensor.
        '''
        Element.__init__(self, name)
        if Xc is None: Xc = zeros(3)
        if inertia is None: inertia = zeros((3,3))
        self.mass = mass
        self.inertia = inertia
        self.Xc = Xc
        # Set constant parts of mass matrix
        self.mass_vv[VP,VP] = self.mass * eye(3)

    def shape(self):
        return [
            np.r_[ [self.rp], [self.rp+dot(self.Rp,self.Xc)] ]
        ]

    shape_plot_options = [
        {'marker': 'x', 'ms': 10, 'c': 'y', 'mew': 3},
    ]

    def calc_mass(self):
        # global offset to centre of mass
        xc = dot(self.Rp, self.Xc)
        Jp = dot(self.Rp, dot(self.inertia, self.Rp.T))
        wps = skewmat(self.vp[3:])

        ## MASS MATRIX ##
        #    mass_vv[VP,VP] constant
        self.mass_vv[VP,WP] = -self.mass * skewmat(xc)
        self.mass_vv[WP,VP] =  self.mass * skewmat(xc)
        self.mass_vv[WP,WP] =  Jp

        ## QUADRATIC FORCES ## (remaining terms)
        self.quad_forces[VP] = self.mass * dot(dot(wps,wps), xc)
        self.quad_forces[WP] = dot(wps, dot(Jp, self.vp[3:]))

# Slices to refer to parts of matrices
VP = slice(0,3)
WP = slice(3,6)
VD = slice(6,9)
WD = slice(9,12)

class UniformBeam(Element):
    _ndistal = 1
    _nstrain = 6
    _nconstraints = NQD

    def __init__(self, name, length, density, EA, EIy, EIz, GIx=0.0, Jx=0.0, wind=None):
        '''
        Euler beam element.

         - length  : undeformed length of beam
         - density : mass per unit length
         - EA      : Extensional stiffness
         - EIy     : bending stiffness about y axis
         - EIz     : bending stiffness about z axis
         - GIx     : torsional stiffness (G*Ix*kx)

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
        self.wind = wind
        self._initial_calcs()

    def _calc_mass_coeffs(self):
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
#        self._mass_coeffs = array([
#            [13*l0*m/35, -l0**2*m/105, 9*l0*m/70, 13*l0**2*m/420],
#            [0, 2*l0**3*m/105, -31*l0**2*m/420, -l0**3*m/84],
#            [0, 0, 13*l0*m/35, 11*l0**2*m/210],
#            [0, 0, 0, l0**3*m/105]
#        ])

    def _initial_calcs(self):
        self._calc_mass_coeffs()
        # Set constant parts of mass matrix
        c = self._mass_coeffs
        I = eye(3)
        self.mass_vv[VP,VP] = c[0,0]*I
        self.mass_vv[VP,VD] = c[0,2]*I
        self.mass_vv[VD,VP] = c[0,2]*I
        self.mass_vv[VD,VD] = c[2,2]*I

        # Shape functions at distal node
        self.Sr_d = self._beam_rfuncs(1.0)
        self.Sq_d = self._beam_qfuncs(1.0)

        # Stiffness matrix
        EA = self.EA
        GIx,EIy,EIz = self.GIx, self.EIy, self.EIz
        l = self.length
        self.stiffness = array([
            [EA, 0,           0,           0,     0,          0         ],
            [0,  12*EIz/l**3, 0,           0,     0,         -6*EIz/l**2],
            [0,  0,           12*EIy/l**3, 0,     6*EIy/l**2, 0,        ],
            [0,  0,           0,           GIx/l, 0,          0         ],
            [0,  0,           6*EIy/l**2,  0,     4*EIy/l,    0         ],
            [0, -6*EIz/l**2,  0,           0,     0,          4*EIz/l   ],
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
        assert not np.isnan(q0)
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
        B = self._beam_qfuncs(s)
        return qrot3(dot(B, xstrain))

    def calc_distal_pos(self):
        X  = array([self.length,0,0]) + self.xstrain[0:3]
        R  = qrot3(0.5*self.xstrain[3:6])
        self.rd[:] = self.rp + dot(self.Rp,X)
        self.Rd[:,:] = dot(self.Rp, R)

    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvv * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """
        X  = array([self.length,0,0]) + self.xstrain[0:3] # Distal pos in local coords
        Xd = self.vstrain[0:3]
        wps = skewmat(self.vp[3:])

        q  = self._beam_rotation(1.0, self.xstrain)
        qd = self._beam_rotation_vel(1.0, self.xstrain, self.vstrain)
        E  = euler_param_E(q)
        Ed = euler_param_E(qd)
        wrel = 2*dot(E,qd)
        assert np.allclose(q [1:]*2, self.xstrain[3:])
        assert np.allclose(qd[1:]*2, self.vstrain[3:])

        # distal vel
        self.F_vp[VP,WP] = -dot(self.Rp,dot(skewmat(X),self.Rp.T))

        # Distal vel due to strain
        self.F_ve[VP,:] = dot(self.Rp, self.Sr_d)
        # Distal ang vel due to strain
        # TODO: is this right?
        self.F_ve[WP,:] = dot(self.Rp, dot(2*E[:,1:], self.Sq_d))

        assert not np.any(np.isnan(self.F_vp))
        assert not np.any(np.isnan(self.F_ve))

        # Quadratic velocity terms
        self.F_v2[VP] = 2*dot(wps,     dot(self.Rp,Xd)) + dot(dot(wps,wps), dot(self.Rp, X   ))
        self.F_v2[WP] = 2*dot(self.Rp, dot(Ed,qd     )) + dot(wps,          dot(self.Rp, wrel))

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

    def calc_mass(self):
        ep = self.Rp[:,0] # unit vectors along elastic line
        ed = self.Rd[:,0]
        eps = skewmat(self.Rp[:,0]) # unit vectors along elastic line
        eds = skewmat(self.Rd[:,0])
        wps = skewmat(self.vp[WP]) # angular velocity matrices
        wds = skewmat(self.vd[WP])

        # lumped inertia of cross-section
        # XXX check this is right - also add perpendicular inertia of cross-
        #     section, assumed to be half (ok for uniform laminar)
        Jbar = self.Jx * np.diag([1.0, 0.5, 0.5])
        Jxxp = dot(self.Rp, dot(Jbar, self.Rp.T))
        Jxxd = dot(self.Rd, dot(Jbar, self.Rd.T))

        # shorthand
        c = self._mass_coeffs
        m = self.mass_vv
        gv = self.quad_forces

        ## MASS MATRIX ##
        #[VP,VP] constant
        m[VP,WP] = -c[0,1]*eps
        #[VP,VD] constant
        m[VP,WD] = -c[0,3]*eds

        m[WP,VP] =  m[VP,WP].T
        m[WP,WP] = -c[1,1]*dot(eps,eps) + Jxxp
        m[WP,VD] =  c[1,2]*eps
        m[WP,WD] = -c[1,3]*dot(eps,eds)

        #[VD,VP] constant
        m[VD,WP] =  m[WP,VD].T
        #[VD,VD] constant
        m[VD,WD] = -c[2,3]*eds

        m[WD,VP] =  m[VP,WD].T
        m[WD,WP] =  m[WP,WD].T
        m[WD,VD] =  m[VD,WD].T
        m[WD,WD] = -c[3,3]*dot(eds,eds) + Jxxd

        ## QUADRATIC FORCES ## (remaining terms)
        gv[VP] = c[0,1]*dot(        dot(wps,wps) ,ep) + c[0,3]*dot(        dot(wds,wds) ,ed)
        gv[WP] = c[1,1]*dot(dot(eps,dot(wps,wps)),ep) + c[1,3]*dot(dot(eps,dot(wds,wds)),ed)
        gv[VD] = c[1,2]*dot(        dot(wps,wps) ,ep) + c[2,3]*dot(        dot(wds,wds) ,ed)
        gv[WD] = c[1,3]*dot(dot(eds,dot(wps,wps)),ep) + c[3,3]*dot(dot(eds,dot(wds,wds)),ed)
        # quadratic stresses are all zero

    def calc_external_loading(self):
        # Gravity loads
        self._set_gravity_force()

        # Constitutive loading
        self.applied_stress[:] = dot(self.stiffness, self.xstrain) + \
                                 dot(self.damping, dot(self.stiffness, self.vstrain))

        # Wind loading
        if self.wind:
            Cd = 2
            diameter = 2.0
            # loop through stations
            Nst = 5
            seclength = self.length / Nst
            for xi in np.linspace(0,1,num=Nst):
                # XXX This would depend on position for non-uniform wind
                windvel = self.wind.get_wind_speed(self.system.time)
                local_R = self._beam_R(xi, self.xstrain)
                Xskew = skewmat(self._beam_centreline(xi, self.xstrain))
                SX = self._beam_rfuncs(xi)
                local_windvel = dot(dot(self.Rp,local_R).T, windvel)
                local_dragforce = 0.5*1.225*Cd*diameter*(local_windvel)**2 * seclength
                # generalised forces
                Qrp = dot(dot(self.Rp,local_R), local_dragforce)
                Qwp = dot(self.Rp, dot(Xskew, dot(local_R, local_dragforce)))
                Qstrain = dot(SX.T, dot(local_R, local_dragforce))
                self.applied_forces[VP] += Qrp
                self.applied_forces[WP] += Qwp
                self.applied_stress[:]  += Qstrain

#==============================================================================
# class EulerBeam(UniformBeam):
#     def __init__(self, name, x, density, EA, EIy, EIz, GIx=0.0, Jx=0.0, wind=None):
#         '''
#         Euler beam element.
#
#          - x       : undeformed coordinates where beam properties are given
#          - density : mass per unit length, specified at points given in x
#          - EA      : Extensional stiffness
#          - EIy     : bending stiffness about y axis
#          - EIz     : bending stiffness about z axis
#          - GIx     : torsional stiffness (G*Ix*kx)
#
#         '''
#         Element.__init__(self, name)
#         assert x[0] == 0.0
#         self.length = x[-1]
#         self.props_x = x
#         self.linear_density = density
#         self.EA = EA
#         self.GIx = GIx
#         self.EIy = EIy
#         self.EIz = EIz
#         self.Jx = Jx
#         self.wind = wind
#         self.damping = 0.1
#         self._initial_calcs()
#
#     def _calc_mass_coeffs(self):
#         rho = self.linear_density # mass per unit length
#         x = self.props_x
#         l = self.props_x[-1]
#
#         # Integrals of interpolating factors
#         #  p1 = 3 xi^2 - 2 xi^3 = (3x^2 * l - 2x^3)/l^3
#         #  p2 =   xi^2 -   xi^3 = ( x^2 * l -  x^3)/l^3
#         p1 = (3 * l * x**2 - 2 * x**3) / l**3
#         p2 = (    l * x**2 -     x**3) / l**3
#
#         # Combined terms
#         terms = array([
#             1 - p1,
#             x - l*(p1 - p2),
#             p1,
#             l*p2,
#         ])
#
#         self._mass_coeffs = np.zeros((4,4))
#         for i in range(4):
#             for j in range(4):
#                self._mass_coeffs[i,j] = np.trapz(rho * terms[i] * terms[j], x)
#==============================================================================

class ModalElement(Element):
    _ndistal = 0
    _nstrain = 6
    _nconstraints = 0

    def __init__(self, name, modes, loading=None):
        '''
        General element represented by mode shapes. Assume no distal nodes.

         - modal_rep : ModalRepresentation instance describing the
                       modal representation

        '''
        self._nstrain = len(modes.freqs)
        Element.__init__(self, name)
        self.modes = modes
        self.loading = loading

        # Set constant parts of mass matrix
        self.mass_vv[VP,VP] = self.modes.mass * eye(3)
        self.mass_ee[ :, :] = self.modes.strain_strain()

        # Stiffness matrix
        self.stiffness = np.diag(self.mass_ee) * self.modes.freqs**2
        self.damping = 2 * self.modes.damping * self.stiffness / self.modes.freqs

    def station_positions(self):
        prox_pos = self.modes.X(self.xstrain)
        global_pos = self.rp[None,:] + np.einsum('ij,hj', self.Rp, prox_pos)
        return global_pos
    
    def station_rotations(self):
        prox_rot = self.modes.R(self.xstrain)
        global_rot = np.einsum('ij,hjk', self.Rp, prox_rot)
        return global_rot
    
    def station_velocities(self):
        prox_pos = self.modes.X(self.xstrain)
        prox_vel = self.modes.Xdot(self.vstrain)
        global_vel = self.vp[None,VP] + np.einsum('ij,hj', self.Rp, prox_vel) \
            + np.einsum('ijk,j,kl,hl', eps_ijk, self.vp[WP], self.Rp, prox_pos)
        return global_vel

    def shape(self):
        # proximal values
        X0 = self.modes.X0
        defl = dot(self.modes.shapes, self.xstrain)
        shape = [dot(self.Rp, (X0[i]+defl[i])) for i in range(X0.shape[0])]
        return [
            self.rp + array(shape),
            np.r_[ [self.rp], [self.rp + dot(self.Rp, X0[-1])], [shape[-1]] ]
        ]

    shape_plot_options = [
        {'c': 'g', 'marker': 'o', 'lw': 2, 'ms': 1},
        {'c': 'k', 'lw': 2, 'alpha': 0.3},
    ]

    def calc_mass(self):
        # angular velocity skew matrix
        wp = self.vp[WP]
        wps = skewmat(wp)
        local_wp = dot(self.Rp.T, wp)

        # Inertia tensor made up of undefomed inertia J0, and contributions
        # from shapes
        inertia = self.modes.inertia_tensor(self.xstrain)
        inertia_global = dot(self.Rp, dot(inertia, self.Rp.T))

        # Linear-rotation term
        rw_global = dot(self.Rp, self.modes.I0 + dot(self.modes.S, self.xstrain))
        rw_global_skew = skewmat(rw_global)

        # Rotation-strain term
        wf_global = dot(self.Rp, self.modes.rotation_strain(self.xstrain))

        # 1st shape int in global coords
        S_global = dot(self.Rp, self.modes.S)

        ## MASS MATRIX ##
        #    mass_vv[VP,VP] constant
        self.mass_vv[VP,WP] = -rw_global_skew
        self.mass_vv[WP,VP] =  rw_global_skew
        self.mass_vv[WP,WP] =  inertia_global
        self.mass_ve[VP, :] =  S_global
        self.mass_ve[WP, :] =  wf_global
        # mass_ee constant

        ## QUADRATIC FORCES ## (remaining terms)

        # Centrifugal forces
        centrifugal = dot(dot(wps,wps), rw_global)

        # Force dependent on strain velocity
        strainvel = 2*dot(wps, dot(self.Rp, dot(self.modes.S, self.vstrain)))

        # Terms for moments dependent on strain velocity
        ang_strainvel_local = self.modes.inertia_vel(self.xstrain, self.vstrain)
        ang_strainvel_global = dot(self.Rp, dot(ang_strainvel_local, self.Rp.T))

        self.quad_forces[VP] = centrifugal + strainvel
        self.quad_forces[WP] = dot(
            (dot(wps, inertia_global) + 2*ang_strainvel_global), wp)
        self.quad_stress[ :] = self.modes.quad_stress(self.xstrain, self.vstrain, local_wp)

    def _get_loading(self):
        assert self.loading is not None
        # Positions and orientations of all stations
        prox_rot = self.modes.R(self.xstrain)
        global_pos = self.station_positions() #self.rp[None,:] + np.einsum('ij,hj', self.Rp, prox_pos)
        global_rot = self.station_rotations() #np.einsum('ij,hjk', self.Rp, prox_rot)
        global_vel = self.station_velocities()
        P_station = self.loading(self.system.time, global_pos,
                                 global_rot, global_vel)
        # P is in local stations coords -- transform back to prox frame
        P_prox = np.einsum('hji,hj->hi', prox_rot, P_station)
        return P_prox

    def calc_external_loading(self):
        # Gravity loads
        self._set_gravity_force()

        # Constitutive loading
        self.applied_stress[:] = (
            self.stiffness * self.xstrain +
            self.damping   * self.vstrain
        )
        
        # External loading
        if self.loading is not None:
            # Positions and orientations of all stations
            P_prox = self._get_loading()
            Fext,Mext,Sext = self.modes.distributed_loading(P_prox, self.xstrain)
            self.applied_forces[VP] += Fext
            self.applied_forces[WP] += Mext
            self.applied_stress[:]  += Sext

class DirectModalElement(Element):
    _ndistal = 0
    _nstrain = 0
    _nconstraints = 0

    def __init__(self, name, modes, loading=None):
        '''
        General element represented by mode shapes. Assume no distal nodes.
        Don't do anything fancy, directly integrate the density distribution
        each time its required -- use as a check on the more efficient modal
        element which uses precalculated shape integrals.

         - modes : ModalRepresentation instance describing the mode shapes

        '''
        self._nstrain = len(modes.freqs)
        Element.__init__(self, name)
        self.modes = modes
        self.loading = loading

        # Calculate mass matrix initially
        self.calc_mass()

        # Stiffness matrix
        self.stiffness = np.diag(self.mass_ee) * self.modes.freqs**2
        self.damping = 2 * self.modes.damping * self.stiffness / self.modes.freqs

    def station_positions(self):
        prox_pos = self.modes.X(self.xstrain)
        global_pos = self.rp[None,:] + np.einsum('ij,hj', self.Rp, prox_pos)
        return global_pos
    
    def station_rotations(self):
        prox_rot = self.modes.R(self.xstrain)
        global_rot = np.einsum('ij,hjk', self.Rp, prox_rot)
        return global_rot
    
    def station_velocities(self):
        prox_pos = self.modes.X(self.xstrain)
        prox_vel = self.modes.Xdot(self.vstrain)
        global_vel = self.vp[None,VP] + np.einsum('ij,hj', self.Rp, prox_vel) \
            + np.einsum('ijk,j,kl,hl', eps_ijk, self.vp[WP], self.Rp, prox_pos)
        return global_vel

    def shape(self):
        # proximal values
        X0 = self.modes.X0
        defl = dot(self.modes.shapes, self.xstrain)
        shape = [dot(self.Rp, (X0[i]+defl[i])) for i in range(X0.shape[0])]
        return [
            self.rp + array(shape),
            np.r_[ [self.rp], [self.rp + dot(self.Rp, X0[-1])], [shape[-1]] ]
        ]

    shape_plot_options = [
        {'c': 'g', 'marker': 'o', 'lw': 2, 'ms': 1},
        {'c': 'k', 'lw': 2, 'alpha': 0.3},
    ]

    def calc_mass(self):
        # angular velocity skew matrix
        wp = self.vp[WP]
        wps = skewmat(wp)
        
        # Rp * shapes
        RU = np.einsum('ij,hjp', self.Rp, self.modes.shapes)
        
        # station local positions and global positions
        X = self.modes.X(self.xstrain)
        x = np.einsum('ij,hj', self.Rp, X)

        M = np.zeros((len(self.modes.x),6+len(self.modes.freqs),6+len(self.modes.freqs)))
        g = np.zeros((len(self.modes.x),6+len(self.modes.freqs)))

        # Step through stations and work out integrands
        for i in range(M.shape[0]):
            xs = skewmat(x[i])
            G = dot(wps, dot(wps, x[i]) + 2*dot(RU[i], self.vstrain))
            
            M[i,VP,VP] =  np.eye(3)
            M[i,WP,WP] = -dot(xs, xs)
            M[i,6:,6:] = dot(RU[i].T, RU[i])
            
            M[i,VP,WP] = -xs
            M[i,WP,VP] =  xs
            M[i,VP,6:] = RU[i]
            M[i,6:,VP] = RU[i].T
            M[i,WP,6:] = dot(xs, RU[i])
            M[i,6:,WP] = dot(xs, RU[i]).T
            
            g[i,VP] = G
            g[i,WP] = dot(xs, G)
            g[i,6:] = dot(RU[i].T, G)

        # Integrate
        Mint = simps(M * self.modes.density[:,None,None], self.modes.x, axis=0)
        gint = simps(g * self.modes.density[:,None],      self.modes.x, axis=0)
        
        # Save
        self.mass_vv[:,:] = Mint[:6,:6]
        self.mass_ve[:,:] = Mint[:6,6:]
        self.mass_ee[:,:] = Mint[6:,6:]
        self.quad_forces[:] = gint[:6]
        self.quad_stress[:] = gint[6:]

    def _get_loading(self):
        assert self.loading is not None
        # Positions and orientations of all stations
        prox_rot = self.modes.R(self.xstrain)
        global_pos = self.station_positions() #self.rp[None,:] + np.einsum('ij,hj', self.Rp, prox_pos)
        global_rot = self.station_rotations() #np.einsum('ij,hjk', self.Rp, prox_rot)
        global_vel = self.station_velocities()
        P_station = self.loading(self.system.time, global_pos,
                                 global_rot, global_vel)
        # P is in local stations coords -- transform back to prox frame
        P_prox = np.einsum('hji,hj->hi', prox_rot, P_station)
        return P_prox

    def calc_external_loading(self):
        # Gravity loads
        self._set_gravity_force()

        # Constitutive loading
        self.applied_stress[:] = (
            self.stiffness * self.xstrain +
            self.damping   * self.vstrain
        )
        
        # External loading
        if self.loading is not None:
            # Positions and orientations of all stations
            P_prox = self._get_loading()
            Fext,Mext,Sext = self.modes.distributed_loading(P_prox, self.xstrain)
            self.applied_forces[VP] += Fext
            self.applied_forces[WP] += Mext
            self.applied_stress[:]  += Sext

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

##################
###  Outputs   ###
##################

class NodeOutput(object):
    def __init__(self, state_name, deriv=0, local=False, label=None):
        assert deriv in (0,1,2)
        self.state_name = state_name
        self.deriv = deriv
        self.local = local      
        self.label = label
    
    def __str__(self):
        if self.label is not None:
            return self.label
        s = "node <{}>".format(self.state_name)
        if self.deriv == 1: s = "d/dt " + s
        if self.deriv == 2: s = "d2/dt2 " + s
        if self.local: s += " [local]"
        return s
        
    def __call__(self, system):
        if   self.deriv == 0: q = system.q
        elif self.deriv == 1: q = system.qd
        elif self.deriv == 2: q = system.qdd
        assert q.get_type(self.state_name) in ('node','ground')
        output = q[self.state_name]
        
        if self.local:
            if self.deriv == 0:
                raise NotImplemented("What does that mean?")
            else:
                assert len(output) == NQD
                R = system.q[self.state_name][3:].reshape((3,3))
                v = dot(R.T, output[:3])
                w = dot(R.T, output[3:])
                output = np.r_[ v, w ]
                
        return output

class StrainOutput(object):
    def __init__(self, state_name, deriv=0, label=None):
        assert deriv in (0,1,2)
        self.state_name = state_name
        self.deriv = deriv
        self.label = label
    
    def __str__(self):
        if self.label is not None:
            return self.label
        s = "strain <{}>".format(self.state_name)
        if self.deriv == 1: s = "d/dt " + s
        if self.deriv == 2: s = "d2/dt2 " + s
        return s
        
    def __call__(self, system):
        if   self.deriv == 0: q = system.q
        elif self.deriv == 1: q = system.qd
        elif self.deriv == 2: q = system.qdd
        assert q.get_type(self.state_name) in ('strain')
        output = q[self.state_name]
        return output

class LoadOutput(object):
    def __init__(self, state_name, local=False, label=None):
        self.state_name = state_name
        self.local = local      
        self.label = label
    
    def __str__(self):
        if self.label is not None:
            return self.label
        s = "reaction load on <{}>".format(self.state_name)
        if self.local: s += " [local]"
        return s
        
    def __call__(self, system):
        assert system.joint_reactions.get_type(self.state_name) in ('node','ground')
        output = system.joint_reactions[self.state_name]
        
        if self.local:
            assert len(output) == NQD
            R = system.q[self.state_name][3:].reshape((3,3))
            #print '{} transforming to local coords:'.format(self)
            #print R
            v = dot(R.T, output[:3])
            w = dot(R.T, output[3:])
            output = np.r_[ v, w ]
                
        return output
        
class CustomOutput(object):
    transformable = False
    
    def __init__(self, func, label=None):
        self.func = func
        self.label = label
        
    def __str__(self):
        if self.label is not None:
            return self.label
        return "Custom output <{}>".format(self.func)
        
    def __call__(self, system):
        return np.atleast_1d(self.func(system))

class Integrator(object):
    """
    Solve and integrate a system, keeping track of which outputs are required.
    """
    def __init__(self, system, outputs=('pos',), method='dopri5'):
        self.system = system
        self.t = np.zeros(0)
        self.y = []
        self.method = method
        
        # By default, output DOFs
        self._outputs = []
        idof = self.system.qd.names_by_type('strain')
        if 'pos' in outputs:
            for i in idof: self.add_output(StrainOutput(i, deriv=0))
        if 'vel' in outputs:
            for i in idof: self.add_output(StrainOutput(i, deriv=1))
        if 'acc' in outputs:
            for i in idof: self.add_output(StrainOutput(i, deriv=2))
    
    def add_output(self, output):
        self._outputs.append(output)
        
    def outputs(self):
        return [output(self.system) for output in self._outputs]
    
    def labels(self):
        return [str(output) for output in self._outputs]
        
    
    
    def integrate(self, tvals, dt=None, nprint=20):
        if not np.iterable(tvals):
            assert dt is not None
            tvals = np.arange(0, tvals, dt)
        
        # prepare for first outputs
        self.system.update_kinematics(tvals[0]) # work out kinematics
        self.system.solve_reactions()
                
        initial_outputs = self.outputs()
        self.t = tvals
        self.y = []
        for y0 in initial_outputs:
            self.y.append(zeros((len(tvals),len(y0))))
            self.y[-1][0] = y0
        
        iDOF_q  = self.system.q.indices_by_type('strain')
        iDOF_qd = self.system.qd.indices_by_type('strain')
        assert len(iDOF_q) == len(iDOF_qd)
        nDOF = len(iDOF_q)
                
        # Gradient function
        def _func(ti, yi):
            # update system state with states and state rates
            # y constains [strains, d(strains)/dt]
            self.system.q [iDOF_q]  = yi[:nDOF]
            self.system.qd[iDOF_qd] = yi[nDOF:]
            self.system.update_kinematics(ti) # kinematics and dynamics
    
            # solve system
            new_qdd = self.system.solve_accelerations()
            
            #print self.system.qdd[iDOF_qd][2]
            
            # new state vector is [strains_dot, strains_dotdot]
            return np.r_[ self.system.qd[iDOF_qd], new_qdd[iDOF_qd] ]

        # Initial conditions
        integrator = ode(_func)
        integrator.set_integrator(self.method)
        z0 = np.r_[ self.system.q[iDOF_q], self.system.qd[iDOF_qd] ]
        integrator.set_initial_value(z0, tvals[0])        
        
        if nprint is not None:
            print 'Running simulation:',
            sys.stdout.flush()
            tstart = time.clock()
        
        for it,t in enumerate(tvals[1:], start=1):
            integrator.integrate(t)
            if not integrator.successful():
                print 'stopping'
                break
            
            #print self.system.qd['hinge_oop-strains'][0], '\t', integrator.y[11]
            
            # Update kinematics with new timestep's strains and strain rates,
            # and calculate reaction forces
            assert np.all(self.system.q[iDOF_q] == integrator.y[:nDOF])
            assert np.all(self.system.qd[iDOF_qd] == integrator.y[nDOF:])
            self.system.q [iDOF_q]  = integrator.y[:nDOF]
            self.system.qd[iDOF_qd] = integrator.y[nDOF:]
            #self.system.update_kinematics(t) #XXX , calculate_matrices=False) # don't need mass
            self.system.solve_reactions()
            
            # Save outputs
            for y,out in zip(self.y, self.outputs()):
                y[it] = out
            if nprint is not None and (it % nprint) == 0:
                sys.stdout.write('.'); sys.stdout.flush()
        
        if nprint is not None:
            print 'done (%.1f seconds)' % (time.clock() - tstart)

        return self.t, self.y