"""
Base Element class
"""

import numpy as np
from numpy import zeros, ones, eye, dot
from ..utils import update_skewmat, skewmat

# XXX
gravity = 9.81
OPT_GRAVITY = True


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
        self.wps = zeros((3,3))

        # Default [constant] parts of transformation matrices:
        # distal node velocities are equal to proximal, no strain effects
        # acceleration constraint -Fv2 = [ prox->dist, -I, strain->dist ] * [ a_prox, a_dist, a_strain ]
        self.F_vp = np.tile(eye(6), (self._ndistal,1))
        self.F_vd = -eye(6)
        self.F_ve = zeros((6*self._ndistal, self._nstrain))
        self.F_v2 = zeros( 6*self._ndistal                )

        # Default [constant] parts of mass matrices
        nnodes = 6 * (1 + self._ndistal)
        self.mass_vv = zeros((nnodes,nnodes))
        self.mass_ve = zeros((nnodes,self._nstrain))
        self.mass_ee = zeros((self._nstrain,self._nstrain))
        self.quad_forces = zeros(6*(1 + self._ndistal))
        self.quad_stress = zeros(self._nstrain)

        # External forces and stresses
        self.applied_forces = zeros(6 * (1 + self._ndistal))
        self.applied_stress = zeros(self._nstrain)

        # Gravity acceleration
        self._gravacc = np.tile([0, 0, -gravity * OPT_GRAVITY, 0, 0, 0], 1 + self._ndistal)

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

        self.system.joint_reactions[self.system.qd.named_indices[self.iprox]] \
            += np.r_[ Fprox, Mprox ]

    def _set_gravity_force(self):
        self.applied_forces[:] = dot(self.mass_vv, self._gravacc)

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
