# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 17:24:13 2012

@author: Rick Lupton
"""

from __future__ import division
import time
import sys
import operator
import itertools
import numpy as np
from numpy import array, zeros, eye, dot, pi, cos, sin
import numpy.linalg as LA
import scipy.linalg
import scipy.optimize
from scipy.integrate import ode

from . import assemble
from .utils import update_skewmat, skewmat


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
            self.calc_projections()
            # eval residue: Q includes external forces, quadratic
            # terms and internal forces...
            Q = self.rhs[self.iReal]
            # ... and ma includes inertial loads
            qdd = self.Sc  # prescribed accelerations
            ma = dot(self.lhs[np.ix_(self.iReal, self.iReal)], qdd)
            return dot(self.R.T, (Q - ma))

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


####################################################################
####################################################################


class Element(object):
    _ndistal = 0
    _nstrain = 0
    _nconstraints = 0

    def __init__(self, name):
        self.name = name
        self.children = [[]] * self._ndistal
        self.system = None

        self.rp = zeros(3)
        self.Rp = eye(3)
        self.rd = np.tile(zeros(3), self._ndistal)
        self.Rd = np.tile(eye(3), self._ndistal)
        self.xstrain = zeros(self._nstrain)

        self.vp = zeros(6)
        self.vd = zeros(6*self._ndistal)
        self.vstrain = zeros(self._nstrain)

        self.ap = zeros(6)
        self.ad = zeros(6*self._ndistal)
        self.astrain = zeros(self._nstrain)

        # The skew matrix of the prox angular velocity is used a lot - keep
        # it preallocated
        self.wps = zeros((3, 3))

        # Default [constant] parts of transformation matrices:
        # distal node velocities are equal to proximal, no strain effects
        # acceleration constraint:
        # -Fv2 = [prox->dist, -I, strain->dist] * [a_prox, a_dist, a_strain]
        self.F_vp = np.tile(eye(6), (self._ndistal, 1))
        self.F_vd = -eye(6)
        self.F_ve = zeros((6*self._ndistal, self._nstrain))
        self.F_v2 = zeros(6*self._ndistal)

        # Default [constant] parts of mass matrices
        nnodes = 6 * (1 + self._ndistal)
        self.mass_vv = zeros((nnodes, nnodes))
        self.mass_ve = zeros((nnodes, self._nstrain))
        self.mass_ee = zeros((self._nstrain, self._nstrain))
        self.quad_forces = zeros(6*(1 + self._ndistal))
        self.quad_stress = zeros(self._nstrain)

        # External forces and stresses
        self.applied_forces = zeros(6 * (1 + self._ndistal))
        self.applied_stress = zeros(self._nstrain)

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<%s "%s">' % (self.__class__.__name__, self.name)

    def add_leaf(self, elem, distnode=0):
        self.children[distnode].append(elem)

    def iter_leaves(self):
        for node_children in self.children:
            for child in node_children:
                yield child
                for descendent in child.iter_leaves():
                    yield descendent

    def print_info(self):
        print('{!r}:'.format(self))
        print('    prox node: {}'.format(self.iprox))
        if self._nstrain:
            print('    strains: {} ({})'.format(self._nstrain, self.istrain))
        if self._nconstraints:
            print('    constraints: {} ({})'
                  .format(self._nconstraints, self.imult))
        if self._ndistal:
            print('    distal nodes: {}'.format(', '.join(self.idist)))

    def setup_chain(self, system, prox_indices):
        self.system = system
        self.iprox = prox_indices

        # Request new states for constraint multipliers
        self.imult = system.request_new_constraints(self, self._nconstraints)

        # Request new states for internal strains
        self.istrain = system.request_new_strains(self, self._nstrain)

        # Request new states for distal node coordinates
        self.idist = [system.request_new_node(self)
                      for i in range(self._ndistal)]

        # Pass onto children for each distal node
        for inode, node_children in zip(self.idist, self.children):
            for child in node_children:
                child.setup_chain(system, inode)

    def finish_setup(self):
        system = self.system

        # Setup views into global arrays
        self.rp = system.q[self.iprox][:3]
        self.Rp = system.q[self.iprox][3:].reshape((3, 3))
        self.vp = system.qd[self.iprox]
        self.ap = system.qdd[self.iprox]
        self.xstrain = system.q[self.istrain]
        self.vstrain = system.qd[self.istrain]
        self.astrain = system.qdd[self.istrain]

        assert len(self.idist) <= 1
        for idist in self.idist:
            self.rd = system.q[idist][:3]
            self.Rd = system.q[idist][3:].reshape((3, 3))
            self.vd = system.qd[idist]
            self.ad = system.qdd[idist]

        # Save actual indices as well as names for speed when assembling
        self._iprox = self.system.qd.indices(self.iprox)
        self._idist = sum([self.system.qd.indices(i) for i in self.idist], [])
        self._istrain = self.system.qd.indices(self.istrain)
        self._imult = self.system.qd.indices(self.imult)

        self._set_wrapping()

    def _set_wrapping(self):
        pass

    def calc_external_loading(self):
        self._set_gravity_force()

    def update(self, calculate_matrices=True):
        # Update cached prox angular velocity skew matrix
        update_skewmat(self.wps, self.vp[3:])

        # Calculate kinematic transforms
        self.calc_kinematics()

        # find distal node values in terms of proximal node values
        if self._ndistal > 0:
            self.calc_distal_pos()
            self.vd[:] = (dot(self.F_vp, self.vp) +
                          dot(self.F_ve, self.vstrain))
            self.ad[:] = (dot(self.F_vp, self.ap) +
                          dot(self.F_ve, self.astrain) + self.F_v2)

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
        a_nodal = np.r_[self.ap, self.ad]
        elem = (-dot(self.mass_vv, a_nodal) +
                -dot(self.mass_ve, self.astrain) +
                self.quad_forces +
                self.applied_forces)

        # calculate forces acting ON proximal nodes:
        #   Fprox + elem_forces_on_prox + sum of distal_forces = 0
        Fprox = -elem[0:3].copy()
        Mprox = -elem[3:6].copy()
        for i, dist in enumerate(self.idist):
            # forces acting ON distal nodes
            distal_forces = (elem[6+6*i:12+6*i] -
                             self.system.joint_reactions[dist])
            relcross = skewmat(self.rd[3*i:3*i+3] - self.rp)
            Fprox += -distal_forces[:3]
            Mprox += -distal_forces[3:]-dot(relcross, distal_forces[:3])

        self.system.joint_reactions[self.system.qd.named_indices[self.iprox]] \
            += np.r_[Fprox, Mprox]

    def gravity_acceleration(self):
        if self.system:
            g = self.system.gravity
        else:
            g = 0
        return array([0, 0, -g, 0, 0, 0])

    def _set_gravity_force(self):
        # Gravity acceleration
        acc = np.tile(self.gravity_acceleration(), 1 + self._ndistal)
        self.applied_forces[:] = dot(self.mass_vv, acc)

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


##################
###  Outputs   ###
##################

class NodeOutput(object):
    def __init__(self, state_name, deriv=0, local=False, label=None):
        assert deriv in (0, 1, 2)
        self.state_name = state_name
        self.deriv = deriv
        self.local = local
        self.label = label

    def __str__(self):
        if self.label is not None:
            return self.label
        s = "node <{}>".format(self.state_name)
        if self.deriv == 1:
            s = "d/dt " + s
        if self.deriv == 2:
            s = "d2/dt2 " + s
        if self.local:
            s += " [local]"
        return s

    def __call__(self, system):
        if self.deriv == 0:
            q = system.q
        elif self.deriv == 1:
            q = system.qd
        elif self.deriv == 2:
            q = system.qdd
        assert q.get_type(self.state_name) in ('node', 'ground')
        output = q[self.state_name]

        if self.local:
            if self.deriv == 0:
                raise NotImplementedError("What does that mean?")
            else:
                assert len(output) == NQD
                R = system.q[self.state_name][3:].reshape((3, 3))
                v = dot(R.T, output[:3])
                w = dot(R.T, output[3:])
                output = np.r_[v, w]

        return output


class StrainOutput(object):
    def __init__(self, state_name, deriv=0, label=None):
        assert deriv in (0, 1, 2)
        self.state_name = state_name
        self.deriv = deriv
        self.label = label

    def __str__(self):
        if self.label is not None:
            return self.label
        s = "strain <{}>".format(self.state_name)
        if self.deriv == 1:
            s = "d/dt " + s
        if self.deriv == 2:
            s = "d2/dt2 " + s
        return s

    def __call__(self, system):
        if self.deriv == 0:
            q = system.q
        elif self.deriv == 1:
            q = system.qd
        elif self.deriv == 2:
            q = system.qdd
        assert q.get_type(self.state_name) in ('strain', 'constraint')
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
        if self.local:
            s += " [local]"
        return s

    def __call__(self, system):
        assert (system.joint_reactions.get_type(self.state_name)
                in ('node', 'ground'))
        output = system.joint_reactions[self.state_name]

        if self.local:
            assert len(output) == NQD
            R = system.q[self.state_name][3:].reshape((3, 3))
            v = dot(R.T, output[:3])
            w = dot(R.T, output[3:])
            output = np.r_[v, w]

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
            for i in idof:
                self.add_output(StrainOutput(i, deriv=0))
        if 'vel' in outputs:
            for i in idof:
                self.add_output(StrainOutput(i, deriv=1))
        if 'acc' in outputs:
            for i in idof:
                self.add_output(StrainOutput(i, deriv=2))

    def add_output(self, output):
        self._outputs.append(output)

    def outputs(self):
        return [output(self.system) for output in self._outputs]

    @property
    def labels(self):
        return [str(output) for output in self._outputs]

    def integrate(self, tvals, dt=None, t0=0.0, callback=None,
                  extra_states=None, nprint=20):

        if not np.iterable(tvals):
            assert dt is not None and t0 is not None
            tvals = np.arange(0.0, tvals, dt)
        out_tvals = tvals[tvals >= t0]
        it0 = np.nonzero(tvals >= t0)[0][0]

        if extra_states is None:
            extra_states = zeros(0)

        # prepare for first outputs
        #self.system.update_kinematics(0.0) # work out kinematics
        #self.system.solve_reactions()

        self.t = out_tvals
        self.y = []
        for y0 in self.outputs():
            self.y.append(zeros((len(out_tvals),) + y0.shape))
        if len(extra_states) > 0:
            self.y.append(zeros((len(out_tvals), len(extra_states))))

        iDOF_q  = self.system.q.indices_by_type('strain')
        iDOF_qd = self.system.qd.indices_by_type('strain')
        assert len(iDOF_q) == len(iDOF_qd)
        nStruct = len(iDOF_q)
        nOther = len(extra_states)

        # Gradient function
        def _func(ti, yi):
            # update system state with states and state rates
            # y constains [other, strains, d(strains)/dt]
            q_other = yi[:nOther]
            self.system.q[iDOF_q] = yi[nOther:nOther+nStruct]
            self.system.qd[iDOF_qd] = yi[nOther+nStruct:]
            self.system.update_kinematics(ti, calculate_matrices=False)

            # Callback may e.g. set element loading
            if callback:
                q_struct = self.system.get_state()
                qd_other = callback(self.system, ti, q_struct, q_other)
                assert len(qd_other) == nOther
            else:
                qd_other = []

            # solve system
            self.system.update_kinematics(ti, calculate_matrices=True)
            self.system.solve_accelerations()

            # new state vector is [other_dot, strains_dot, strains_dotdot]
            return np.r_[
                qd_other,
                self.system.qd[iDOF_qd],
                self.system.qdd[iDOF_qd]
            ]

        # Initial conditions
        self.system.set_initial_velocities(time=0.0)
        z0 = np.r_[
            extra_states,
            self.system.q[iDOF_q],
            self.system.qd[iDOF_qd]
        ]

        # Setup actual integrator
        integrator = ode(_func)
        integrator.set_integrator(self.method)
        integrator.set_initial_value(z0, 0.0)

        if nprint is not None:
            print('Running simulation:',)
            sys.stdout.flush()
            tstart = time.clock()

        # Update for first outputs
        _func(0.0, z0)

        try:
            for it, t in enumerate(tvals):
                if t > 0:
                    integrator.integrate(t)
                    if not integrator.successful():
                        print('stopping')
                        break

                # Wrap joint angles etc
                self.system.q.wrap_states()

                if t >= t0:
                    # Update nodal accelerations from DOFs' accelerations
                    self.system.update_kinematics()

                    # Calculate reaction forces (backwards iteration down tree)
                    self.system.solve_reactions()

                    # Save outputs
                    for y, out in zip(self.y, self.outputs()):
                        y[it - it0] = out
                    if len(extra_states) > 0:
                        self.y[-1][it - it0] = integrator.y[:nOther]
                    if nprint is not None and (it % nprint) == 0:
                        sys.stdout.write('.')
                        sys.stdout.flush()
                else:
                    if nprint is not None and (it % nprint) == 0:
                        sys.stdout.write('-')
                        sys.stdout.flush()

        except KeyboardInterrupt:
            if nprint is not None:
                print('stopping')
        else:
            if nprint is not None:
                print('done')

        if nprint is not None:
            elapsed_time = time.clock() - tstart
            print(' (%.1f seconds, %d%% of simulation time)' %
                  (elapsed_time, 100*elapsed_time/tvals[-1]))

        return self.t, self.y
