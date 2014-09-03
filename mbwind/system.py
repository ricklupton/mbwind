# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:24:13 2012

@author: Rick Lupton
"""

from __future__ import division
import operator
import itertools
import numpy as np
from numpy import zeros, eye, dot, pi, cos, sin
import numpy.linalg as LA
import scipy.linalg
import scipy.optimize

from . import assemble
from .reduced import ReducedSystem


# Number of generalised position and  velocity coordinates
NQ = 12  # 3 spatial plus 9 rotation matrix entries
NQD = 6  # 3 spatial plus 3 angular velocity


class ArrayProxy(object):
    """Delegate getitem and setitem to a reduced part of the target array"""
    def __init__(self, target, subset=None):
        if subset is None:
            subset = list(range(len(target)))
        self.subset = subset
        self._target = target

    def __getitem__(self, index):
        return self._target[self.subset[index]]

    def __setitem__(self, index, value):
        ix = self.subset[index]
        if not np.isscalar(value) and len(self._target[ix]) != len(value):
            raise ValueError('value length does not match index')
        self._target[ix] = value

    def __len__(self):
        return len(self.subset)

    def __contains__(self, item):
        return item in self.subset


class StateArray(object):
    def __init__(self):
        self.reset()

    def _slice(self, index):
        if isinstance(index, str):
            index = self.named_indices[index]
        if isinstance(index, int):
            index = slice(index, index+1)
        return index

    def indices(self, index):
        s = self._slice(index)
        return list(range(s.start, s.stop))

    def __getitem__(self, index):
        if self._array is None:
            raise RuntimeError("Need to call allocate() first")
        return self._array[self._slice(index)]

    def __setitem__(self, index, value):
        if self._array is None:
            raise RuntimeError("Need to call allocate() first")
        ix = self._slice(index)
        if not np.isscalar(value) and len(self._array[ix]) != len(value):
            raise ValueError('value length does not match index')
        self._array[ix] = value

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
        self.wrap_levels = {}
        self.dofs = ArrayProxy([])
        self._array = None

    def wrap_states(self):
        for i, a in self.wrap_levels.items():
            self[i] = self[i] % a

    def new_states(self, owner, type_, num, name=None):
        if self._array is not None:
            raise RuntimeError("Already allocated, call reset()")
        self.owners.extend([owner] * num)
        self.types.extend([type_] * num)
        if name is not None:
            if name in self.named_indices:
                raise KeyError("{} already exists".format(name))
            N = len(self.owners)
            self.named_indices[name] = slice(N-num, N)

    def indices_by_type(self, types):
        "Return all state indices of the given type"
        if not isinstance(types, (list, tuple)):
            types = (types,)
        return [i for i, t in enumerate(self.types) if t in types]

    def names_by_type(self, types):
        "Return all named states of the given type"
        if not isinstance(types, (list, tuple)):
            types = (types,)
        return [name for name, ind in sorted(self.named_indices.items(),
                                             key=operator.itemgetter(1))
                if self.get_type(name) in types]

    def by_type(self, types):
        "Return all elements of the given type"
        index = self.indices_by_type(types)
        return self._array[index]

    def get_type(self, index):
        types = self.types[self._slice(index)]
        if not all([t == types[0] for t in types]):
            raise ValueError("index refers to different types")
        if types:
            return types[0]
        return None


class System(object):
    def __init__(self, gravity=0.0):
        self.gravity = gravity

        # State vectors and joint reaction forces (acting ON proximal node)
        self.q = StateArray()
        self.qd = StateArray()
        self.qdd = StateArray()
        self.joint_reactions = StateArray()

        # Bookkeeping
        self.first_elements = []
        self.time = 0.0
        self._node_counter = 0

        # Prescribe ground node to be fixed by default
        # velocity and acceleration
        # actually (-b,-c), where
        #   b = -\phi_t, i.e. partial derivative of constraint wrt time
        #   c = -\dot{\phi_t} - \dot{phi_q}\dot{q}
        # assuming these constraints relate only to a single DOF, then the
        # second term in c is zero, and c is just the time derivative of b.
        # Either scalars of a callable can be supplied. The callable will be
        # called with the current time at each step.
        self.prescribed_dofs = {}
        self.q_dof = {}
        self.hooks = []

        #### Set up ####
        # Set up first node
        self.ground_ind = self._new_states(None, 'ground', NQD, NQ, 'ground')

    def add_leaf(self, element):
        self.first_elements.append(element)

    def setup(self):
        # Set up first element to use first node as its proximal node,
        # and set up all its children (who will call request_new_states)
        for elem in self.first_elements:
            elem.setup_chain(self, self.ground_ind)

        # Now number of states is known, can size matrices and vectors
        for states in (self.q, self.qd, self.qdd, self.joint_reactions):
            states.allocate()
        N = len(self.qd)
        self.rhs = zeros(N)       # System LHS matrix and RHS vector
        self.lhs = zeros((N, N))
        self._update_indices()

        # Give elements a chance to finish setting up (make views into
        # global vectors)
        self.elements = {}
        for element in self.iter_elements():
            self.elements[element.name] = element
            element.finish_setup()

    def print_states(self):
        print('      Element        Type           ID             Prescribed')
        print('-------------------------------------------------------------')
        named_indices = sorted(self.qd.named_indices.items(),
                               key=operator.itemgetter(1))
        for name, indices in named_indices:
            i = list(range(indices.start, indices.stop))
            if indices.start == indices.stop:
                continue
            if name == 'ground':
                prescribed = [True]
            else:
                prescribed = [(j in self.prescribed_dofs) for j in i]
            pstr = (' * ' if all(prescribed)
                    else '(*)' if any(prescribed)
                    else '   ')
            print('{:>3}-{:<3} {!s:<15}{:<15}{:<20} {}'
                  .format(i[0], i[-1], self.qd.owners[i[0]],
                          self.qd.types[i[0]], name, pstr))

    def iter_elements(self):
        for elem in self.first_elements:
            yield elem
            for elem2 in elem.iter_leaves():
                yield elem2

    def print_info(self):
        print('System:\n')
        for element in self.iter_elements():
            element.print_info()
            print()

    ##################################################################
    ###   Book-keeping   #############################################
    ##################################################################
    def _new_states(self, elem, type_, num, num_q, name):
        # Nodes have different number of entries in q vector than qd and qdd
        self.q.new_states(elem, type_, num_q, name)
        self.qd.new_states(elem, type_, num, name)
        self.qdd.new_states(elem, type_, num, name)
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
            states.dofs.subset = states.indices_by_type('strain')
            for dof in self.prescribed_dofs:
                states.dofs.subset.remove(
                    (states is self.q) and self.q_dof[dof] or dof)

        # prescribed strains
        self.iNotPrescribed = np.ones(len(self.qd), dtype=bool)
        self.iNotPrescribed[0:6] = False  # ground node
        self.iNotPrescribed[list(self.prescribed_dofs)] = False

        # real coords, node and strains
        ir = self.qd.indices_by_type(('ground', 'node', 'strain'))
        self.iReal = np.zeros(len(self.qd), dtype=bool)
        self.iReal[ir] = True


    def get_state(self):
        return np.concatenate([self.q.dofs[:], self.qd.dofs[:]])

    def set_state(self, state):
        N = len(self.q.dofs[:])
        assert len(state) == 2 * N
        self.q.dofs[:] = state[:N]
        self.qd.dofs[:] = state[N:]
        self.update_kinematics()

    def apply_prescribed_accelerations(self, time):
        """Set prescribed accelerations.

        Note that velocity and position are not set here - they must
        be updated by integration of accelerations, or by additional
        knowledge.
        """
        for dof, acc in self.prescribed_dofs.items():
            if callable(acc):
                acc = acc(time)
            self.qdd[dof] = acc

    def update_kinematics(self, time=None, calculate_matrices=True):
        if time is not None:
            self.time = time

        # Update prescribed accelerations (velocity and position are
        # updated by integration)
        self.apply_prescribed_accelerations(self.time)

        # Reset mass and constraint matrices if updating
        if calculate_matrices:
            self.lhs[:] = 0.0
            self.rhs[:] = 0.0

        # Update kinematics
        r0 = self.q[:3]
        R0 = self.q[3:12].reshape((3, 3))
        r0[:] = 0.0
        R0[:, :] = eye(3)
        for element in self.iter_elements():
            element.update(calculate_matrices)

        # Call loading hooks
        for hook in self.hooks:
            hook(self, time)

        # Assemble mass matrices, constraint matrices and RHS vectors
        if calculate_matrices:
            self.assemble()

    def assemble(self):
        assemble.assemble(self.iter_elements(), self.lhs, self.rhs)

    def prescribe(self, element, acc=0.0, part=None):

        """Prescribe the given element's strains with the velocity and
        acceleration constraints given.

        TODO: clarify these equations

        vel = -b = \phi_t, i.e. partial derivative of constraint wrt time
        acc = -c = \dot{\phi_t} + \dot{phi_q}\dot{q}

        The specified DOFs will be removed from the matrices when solving.
        """
        dofs = list(zip(self.q .indices(element.istrain),
                        self.qd.indices(element.istrain)))
        if isinstance(part, list):
            dofs = [dofs[p] for p in part]
        elif part is not None:
            dofs = [dofs[part]]
        for iq, iqd in dofs:
            self.prescribed_dofs[iqd] = acc
            self.q_dof[iqd] = iq
        self._update_indices()

    def free(self, element):
        """
        Remove any constraints on the element's strains
        """
        dofs = self.qd.indices(element.istrain)
        for iqd in dofs:
            try:
                del self.prescribed_dofs[iqd]
                del self.q_dof[iqd]
            except KeyError:
                pass
        self._update_indices()

    def solve_accelerations(self):
        '''Solve for free accelerations, taking account of any prescribed
        accelerations.
        '''

        prescribed_acc_forces = dot(self.lhs[:, ~self.iNotPrescribed],
                                    self.qdd[~self.iNotPrescribed])

        # remove prescribed acceleration entries from mass matrix and RHS
        # add the forces corresponding to prescribed accelerations back in
        M = self.lhs[self.iNotPrescribed, :][:, self.iNotPrescribed]
        b = (self.rhs - prescribed_acc_forces)[self.iNotPrescribed]

        # solve system for accelerations
        a = LA.solve(M, b)

        result = self.qdd._array.copy()
        result[self.iNotPrescribed] = a

        iStrain = self.qd.indices_by_type('strain')
        self.qdd[iStrain] = result[iStrain]

    def solve_reactions(self):
        """Iterate backwards down tree solving for joint reaction forces.
        Assumes the motion has been solved, i.e. used q, qd, and qdd.
        """
        self.joint_reactions[:] = 0.0
        for elem in reversed(list(self.iter_elements())):
            elem.iter_reactions()

    def find_equilibrium(self):
        """Solve static equalibrium problem, using currently set initial
        velocities and accelerations.
        """
        def func(z):
            # Update system and matrices
            self.q.dofs[:] = z
            self.update_kinematics()
            self.update_matrices()
            rsys = ReducedSystem(self)
            return rsys.Q

        q0 = scipy.optimize.fsolve(func, self.q.dofs[:])
        self.q.dofs[:] = q0

    ##############################################
    ### Functions for helping with mode shapes ###
    ##############################################
    def dof_index(self, element, strain_number):
        """Return the DOF number for strain ``strain_number`` of ``element``"""
        strains = self.elements[element]._istrain
        try:
            return self.qd.dofs.subset.index(strains[strain_number])
        except IndexError:
            raise IndexError('Strain number {} out of range ({})'
                             .format(strain_number, len(strains)))
        except ValueError:
            raise ValueError('Strain is prescribed')

    def free_dof_indices(self, element):
        """Return DOF numbers for all free strains of ``element``"""
        def iterfunc():
            try:
                for istrain in itertools.count():
                    yield self.dof_index(element, istrain)
            except (IndexError, ValueError):
                return  # run out of strains, or strain was prescribed
        return list(iterfunc())

    def convert_mbc_dofs_to_blade(self, v, elements, azimuth):
        """Convert MBC DOFs in ``v`` to blade DOFs, at ``azimuth``"""
        if len(elements) != 3:
            raise NotImplementedError("Only 3 blade MBC is implemented")
        vb = v.copy()
        dofs_by_elem = [self.free_dof_indices(element) for element in elements]
        dofs_by_strain = zip(*dofs_by_elem)
        for dofs in dofs_by_strain:                # loop through strains/modes
            for j in range(3):                     # loop through blades
                psi = azimuth + 2*pi*j/3           # azimuth of blade j
                vb[dofs[j]] = (v[dofs[0]] +        # combine MBC back to blade
                               cos(psi)*v[dofs[1]] +
                               sin(psi)*v[dofs[2]])
        return vb
