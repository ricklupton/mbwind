"""
Modal elements
"""

import numpy as np
from numpy import array, zeros, eye, dot, pi
from ..core import Element, eps_ijk

# Slices to refer to parts of matrices
VP = slice(0, 3)
WP = slice(3, 6)
VD = slice(6, 9)
WD = slice(9, 12)


class ModalElement(Element):
    _ndistal = 0
    _nstrain = 6
    _nconstraints = 0

    def __init__(self, name, modes, loading=None, distal=False,
                 damping_freqs=None):
        '''
        General element represented by mode shapes. Assume no distal nodes.

         - modal_rep : ModalRepresentation instance describing the
                       modal representation
         - loading : Loading object describing external loading
         - distal : whether the element needs a distal node
                    (expands size of system by a node)
         - extra_modal_mass : extra mass which has been included in the
             calculation of the modal frequencies, but is not part of the
             element. For example, Bladed adds nacelle and rotor mass to the
             reported tower modal frequencies; to get the correct tower
             stiffness, this has to be taken off again.

         - damping_freqs : reference frequencies used for damping coefficients.
             If None, use modal frequencies. Needed for tower modes, when this
             element wants the isolated modes, but the damping needs the
             combined frequencies in order to match Bladed.

        '''
        self._nstrain = len(modes.freqs)
        if distal:
            self._ndistal = 1
            self._nconstraints = 6

        if damping_freqs is None:
            damping_freqs = modes.freqs

        Element.__init__(self, name)
        self.modes = modes
        self.loading = loading

        # Set constant parts of mass matrix
        self.mass_vv[VP,VP] = self.modes.mass * eye(3)
        self.mass_ee[ :, :] = self.modes.strain_strain()

        # Stiffness matrix
        self.stiffness = np.diag(self.mass_ee) * self.modes.freqs**2
        self.damping = 2 * self.modes.damping * self.stiffness / damping_freqs

        # Geometric stiffness matrix
        # Use int rdm as axial force, multiply by omega**2 later
        self.kG = self.modes.geometric_stiffness(self.modes.I0_dist[:,0])

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

    def calc_distal_pos(self):
        self.rd[:]   = self.station_positions()[-1]
        self.Rd[:,:] = self.station_rotations()[-1]

    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvp * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """
        if self._ndistal == 0:
            return

        # Distal relative position in global coordinates
        xl = dot(self.Rp, self.modes.X(self.xstrain)[-1])

        # distal vel
        self.F_vp[VP,WP] = -skewmat(xl)

        # Distal vel due to strain
        self.F_ve[VP,:] = dot(self.Rp, self.modes.shapes[-1])
        # Distal ang vel due to strain
        self.F_ve[WP,:] = dot(self.Rp, self.modes.rotations[-1])

        # Quadratic velocity terms
        tip_lin_vel = dot(self.modes.shapes[-1],    self.xstrain)
        tip_ang_vel = dot(self.modes.rotations[-1], self.vstrain)
        self.F_v2[VP] = dot(self.wps, dot(self.wps,xl) + 2*dot(self.Rp,tip_lin_vel))
        self.F_v2[WP] = dot(self.wps, dot(self.Rp, tip_ang_vel))

    def calc_mass(self):
#        # angular velocity skew matrix
#        wp = self.vp[WP]
#        #wps = skewmat(wp)
#        local_wp = dot(self.Rp.T, wp)
#
#        # Inertia tensor made up of undefomed inertia J0, and contributions
#        # from shapes
#        inertia = self.modes.inertia_tensor(self.xstrain)
#        #inertia = _modal_calcs.inertia_tensor(self.modes, self.xstrain)
#        inertia_global = dot(self.Rp, dot(inertia, self.Rp.T))
#
#        # Linear-rotation term
#        rw_global = dot(self.Rp, self.modes.I0 + dot(self.modes.S, self.xstrain))
#        rw_global_skew = skewmat(rw_global)
#
#        # Rotation-strain term
#        wf_global = dot(self.Rp, self.modes.rotation_strain(self.xstrain))
#        #wf_global = dot(self.Rp, _modal_calcs.rotation_strain(self.modes, self.xstrain))
#
#        # 1st shape int in global coords
#        S_global = dot(self.Rp, self.modes.S)
#
#        ## MASS MATRIX ##
#        #    mass_vv[VP,VP] constant
#        self.mass_vv[VP,WP] = -rw_global_skew
#        self.mass_vv[WP,VP] =  rw_global_skew
#        self.mass_vv[WP,WP] =  inertia_global
#        self.mass_ve[VP, :] =  S_global
#        self.mass_ve[WP, :] =  wf_global
#        # mass_ee constant

        #test_vv = np.zeros_like(self.mass_vv)
        #test_ve = np.zeros_like(self.mass_ve)
        #test_gv = np.zeros_like(self.quad_forces)
        #test_ge = np.zeros_like(self.quad_stress)
        #_modal_calcs.calc_mass(self.modes, self.Rp, self.vp[WP], self.xstrain,
        #                       self.vstrain, test_vv, test_ve, test_gv, test_ge)
        _modal_calcs.calc_mass(self.modes, self.Rp, self.vp[WP], self.xstrain,
                               self.vstrain, self.mass_vv, self.mass_ve,
                               self.quad_forces, self.quad_stress)

        ## QUADRATIC FORCES ## (remaining terms)

#        # Centrifugal forces
#        centrifugal = dot(dot(self.wps,self.wps), rw_global)
#
#        # Force dependent on strain velocity
#        strainvel = 2*dot(self.wps, dot(self.Rp, dot(self.modes.S, self.vstrain)))
#
#        # Terms for moments dependent on strain velocity
#        ang_strainvel_local = self.modes.inertia_vel(self.xstrain, self.vstrain)
#        #ang_strainvel_local = _modal_calcs.inertia_vel(self.modes, self.xstrain, self.vstrain)
#        ang_strainvel_global = dot(self.Rp, dot(ang_strainvel_local, self.Rp.T))
#
#        #if self.name == 'blade1':
#        #    print dot(self.Rp.T, centrifugal)
#        self.quad_forces[VP] = centrifugal + strainvel
#        self.quad_forces[WP] = dot(
#            (dot(self.wps, inertia_global) + 2*ang_strainvel_global), wp)
#        self.quad_stress[ :] = self.modes.quad_stress(self.xstrain, self.vstrain, local_wp)
#        #self.quad_stress[ :] = _modal_calcs.quad_stress(
#        #                      self.modes, self.xstrain, self.vstrain, local_wp)
#
#        assert np.allclose(test_vv, self.mass_vv)
#        assert np.allclose(test_ve, self.mass_ve)
#        assert np.allclose(test_gv, self.quad_forces)
#        assert np.allclose(test_ge, self.quad_stress)
#

    def _get_loading(self):
        assert self.loading is not None
        # Positions and orientations of all stations in prox. coords
        global_pos = self.station_positions()
        global_vel = self.station_velocities()

        if OPT_BEAM_LOADS_IN_SECTION_COORDS:
            global_rot = self.station_rotations()
        else:
            global_rot = np.tile(self.Rp, (len(self.modes.x),1,1))

        P_station = self.loading(self.system.time, global_pos,
                                 global_rot, global_vel)

        if OPT_BEAM_LOADS_IN_SECTION_COORDS:
            # P is in local stations coords -- transform back to prox frame
            prox_rot = self.modes.R(self.xstrain)
            P_prox = np.einsum('hji,hj->hi', prox_rot, P_station)
        else:
            P_prox = P_station

        return P_prox

    def calc_external_loading(self):
        # Gravity loads
        self.applied_forces[:] = dot(self.mass_vv,   self._gravacc)
        self.applied_stress[:] = -dot(self.mass_ve.T, self._gravacc)
        # XXX NB applied_stresses are negative (so they are as expected for
        #     elastic stresses, but opposite for applied stresses)

        #print self.applied_stress
        #self._set_gravity_force()

        # Constitutive loading (note stiffness and damping are diagonals so * ok)
        self.applied_stress[:] += (
            self.stiffness * self.xstrain +
            self.damping   * self.vstrain
        )

        # Geometric stiffness
        if OPT_GEOMETRIC_STIFFNESS:
            # Calculate magnitude of angular velocity perpendicular to beam
            local_wp_sq = np.sum(dot(self.Rp.T, self.vp[WP])[1:]**2)
            self.applied_stress[:] += local_wp_sq * dot(self.kG, self.xstrain)

        # External loading
        if self.loading is not None:
            # Positions and orientations of all stations
            P_prox = self._get_loading()
            Fext,Mext,Sext = self.modes.distributed_loading(P_prox, self.xstrain)
            self.applied_forces[VP] += Fext
            self.applied_forces[WP] += Mext
            self.applied_stress[:]  += Sext

    # Declare some standard custom outputs
    def output_deflections(self, stations=(-1,)):
        def f(system):
            X = self.modes.X(self.xstrain)
            X[:,0] -= self.modes.x
            return X[stations]
        return CustomOutput(f, label="{} deflections".format(self.name))

    def output_rotations(self, stations=(-1,)):
        def f(system):
            return self.modes.small_rots(self.xstrain)[stations]
        return CustomOutput(f, label="{} rotations".format(self.name))

    def output_positions(self, stations=(-1,)):
        def f(system):
            return self.station_positions()[stations]
        return CustomOutput(f, label="{} positions".format(self.name))

class DistalModalElement(Element):
    # Set in __init__
    _ndistal = None
    _nstrain = None
    _nconstraints = None

    def __init__(self, name, modes, loading=None, distal=False,
                 damping_freqs=None):
        '''
        General element represented by mode shapes. Assume no distal nodes.

         - modal_rep : ModalRepresentation instance describing the
                       modal representation
         - loading : Loading object describing external loading
         - distal : whether the element needs a distal node
                    (expands size of system by a node)
         - extra_modal_mass : extra mass which has been included in the
             calculation of the modal frequencies, but is not part of the element.
             For example, Bladed adds nacelle and rotor mass to the reported
             tower modal frequencies; to get the correct tower stiffness, this
             has to be taken off again.

         - damping_freqs : reference frequencies used for damping coefficients.
             If None, use modal frequencies. Needed for tower modes, when this
             element wants the isolated modes, but the damping needs the
             combined frequencies in order to match Bladed.

        '''
        self._nstrain = len(modes.freqs)
        if distal:
            self._ndistal = 1
            self._nconstraints = NQD
        else:
            self._ndistal = 0
            self._nconstraints = 0
        Element.__init__(self, name)

        if damping_freqs is None:
            damping_freqs = modes.freqs

        self.modes = modes
        self.loading = loading

        self.stiffness = self.modes.stiffness
        self.damping = 2 * self.modes.stiffness * self.modes.damping[None,:] / damping_freqs
        self.damping[np.isnan(self.damping)] = 0.0

        # Geometric stiffness matrix
        # Use int rdm as axial force, multiply by omega**2 later
        self.kG = self.modes.geometric_stiffness(self.modes.I0_dist[:,0])

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

    def calc_distal_pos(self):
        self.rd[:]   = self.station_positions()[-1]
        self.Rd[:,:] = self.station_rotations()[-1]

    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvp * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """
        if self._ndistal == 0:
            return

        # Distal relative position in global coordinates
        xl = dot(self.Rp, self.modes.X(self.xstrain)[-1])

        # distal vel
        self.F_vp[VP,WP] = -skewmat(xl)

        # Distal vel due to strain
        self.F_ve[VP,:] = dot(self.Rp, self.modes.shapes[-1])
        # Distal ang vel due to strain
        self.F_ve[WP,:] = dot(self.Rp, self.modes.rotations[-1])

        # Quadratic velocity terms
        tip_lin_vel = dot(self.modes.shapes[-1],    self.xstrain)
        tip_ang_vel = dot(self.modes.rotations[-1], self.vstrain)
        self.F_v2[VP] = dot(self.wps, dot(self.wps,xl) + 2*dot(self.Rp,tip_lin_vel))
        self.F_v2[WP] = dot(self.wps, dot(self.Rp, tip_ang_vel))

    def shape(self):
        # proximal values
        X0 = self.modes.X0
        defl = dot(self.modes.shapes, self.xstrain)
        shape = [dot(self.Rp, (X0[i]+defl[i])) for i in range(X0.shape[0])]
        return [
            self.rp + array(shape),
            np.r_[ [self.rp], [self.rp + dot(self.Rp, X0[-1])], [self.rp+shape[-1]] ]
        ]

    shape_plot_options = [
        {'c': 'g', 'marker': 'o', 'lw': 2, 'ms': 1},
        {'c': 'k', 'lw': 2, 'alpha': 0.3},
    ]

    def calc_mass(self):
        assert self._ndistal == 1

        wp = self.vp[WP]
        local_wp = dot(self.Rp.T, wp)

        # Inertia tensor made up of undefomed inertia J0, and contributions
        # from shapes
        inertia = self.modes.inertia_tensor(self.xstrain)
        #print self.modes.S1.shape, self.xstrain.shape
        #inertia = _modal_calcs.inertia_tensor(self.modes, self.xstrain)
        inertia_global = dot(self.Rp, dot(inertia, self.Rp.T))

        # Linear-rotation term
        rw_global = dot(self.Rp, self.modes.I0 + dot(self.modes.S, self.xstrain))
        rw_global_skew = skewmat(rw_global)

        #### "RIGID-BODY" part of mass matrix ####
        # This is still constant
        self.mass_vv[VP,VP] = self.modes.mass * eye(3)
        self.mass_vv[VP,WP] = -rw_global_skew
        self.mass_vv[WP,VP] =  rw_global_skew
        self.mass_vv[WP,WP] =  inertia_global

        #### "MODAL" part

        # mapping from element generalised coords to attachment mode displacements
        # and normal mode amplitudes
        q_to_b = zeros((self._nstrain, 2*NQD + self._nstrain))
        #q_to_b[0:3,0:3 ] = -self.Rp.T
        #q_to_b[3:6,3:6 ] = -self.Rp.T
        #q_to_b[0:3,6:9 ] =  self.Rp.T
        #q_to_b[3:6,9:12] =  self.Rp.T
        q_to_b[:,12:] = eye(self._nstrain)

        A = zeros((6, 2*NQD + self._nstrain))
        term1 = self.modes.S
        term2 = np.einsum('ikl,lkp', eps_ijk,
                          self.modes.S1 + np.einsum('klpr,r', self.modes.S2, self.xstrain))
        A[0:3,:] = dot(self.Rp, dot(term1, q_to_b))
        A[3:6,:] = dot(self.Rp, dot(term2, q_to_b))

        P = dot(q_to_b.T, dot(self.modes.strain_strain(), q_to_b))
        P[0:6,:] += A
        P[:,0:6] += A.T

        # This is still constant
        # XXX now I have duplicate attachment strains with no inertia attached
        self.mass_ee[:,:] = P[12:,12:]

        self.mass_vv[:,:] += P[:12,:12]
        self.mass_ve[:,:]  = P[:12,12:]

        ## QUADRATIC FORCES ## (remaining terms)

        #### Coriolis
        S2 = self.modes.S2
        C = (S2[0,2] - S2[2,0])*local_wp[1] + \
            (S2[1,0] - S2[0,1])*local_wp[2] + \
            (S2[2,1] - S2[1,2])*local_wp[0]
        C = 2*dot(q_to_b.T, dot(C, q_to_b))

        Q = np.einsum('ijp,j->ip',
                      self.modes.ss_S1 + np.einsum('ijpr,r', self.modes.ss_S2, self.xstrain),
                      local_wp)
        C[0:3,:] += 2*dot(self.wps, A[0:3,:])
        C[3:6,:] += 2*dot(self.Rp, dot(Q, q_to_b))

        #### Centrifugal
        g = dot(q_to_b.T, -np.einsum('ip,i->p', Q, local_wp))
        g[0:3] += dot(dot(self.wps,self.wps), rw_global)
        g[3:6] += dot(dot(self.wps, inertia_global), wp)

        qdot = np.r_[ self.vp, self.vd, self.vstrain ]
        self.quad_forces[:] = dot(C[:12,:], qdot) + g[:12]
        self.quad_stress[:] = dot(C[12:,:], qdot) + g[12:]

    def _get_loading(self):
        assert self.loading is not None
        # Positions and orientations of all stations in prox. coords
        global_pos = self.station_positions()
        global_vel = self.station_velocities()

        if OPT_BEAM_LOADS_IN_SECTION_COORDS:
            global_rot = self.station_rotations()
        else:
            global_rot = np.tile(self.Rp, (len(self.modes.x),1,1))

        P_station = self.loading(self.system.time, global_pos,
                                 global_rot, global_vel)

        if OPT_BEAM_LOADS_IN_SECTION_COORDS:
            # P is in local stations coords -- transform back to prox frame
            prox_rot = self.modes.R(self.xstrain)
            P_prox = np.einsum('hji,hj->hi', prox_rot, P_station)
        else:
            P_prox = P_station

        return P_prox

    def calc_external_loading(self):
        # Gravity loads
        self.applied_forces[:] = dot(self.mass_vv,   self._gravacc)
        self.applied_stress[:] = -dot(self.mass_ve.T, self._gravacc)
        # XXX NB applied_stresses are negative (so they are as expected for
        #     elastic stresses, but opposite for applied stresses)

        # mapping from element generalised coords to attachment mode displacements
        # and normal mode amplitudes
        q_to_b = zeros((self._nstrain, 6 + self._nstrain))
        q_to_b[0:3,0:3 ] = -self.Rp.T
        q_to_b[3:6,3:6 ] = -self.Rp.T
        q_to_b[0:3,6:9 ] =  self.Rp.T
        q_to_b[3:6,9:12] =  self.Rp.T
        q_to_b[6:,12:] = eye(self._nstrain - 6)

        # Stiffness matrix:
        #  LHS (12 + Nmodes) element generalised coords
        #  RHS ( 6 + Nmodes) relative nodal displacements and rotations + normal modes
        #K = dot(q_to_b.T, self.modes.stiffness)

        #rel_pos =

        # Constitutive loading
        self.applied_stress[:] += (
            dot(self.stiffness, self.xstrain) +
            dot(self.damping,   self.vstrain)
        )

        # Geometric stiffness
        if OPT_GEOMETRIC_STIFFNESS:
            # Calculate magnitude of angular velocity perpendicular to beam
            local_wp_sq = np.sum(dot(self.Rp.T, self.vp[WP])[1:]**2)
            self.applied_stress[:] += local_wp_sq * dot(self.kG, self.xstrain)

        # External loading
        if self.loading is not None:
            # Positions and orientations of all stations
            P_prox = self._get_loading()
            Fext,Mext,Sext = self.modes.distributed_loading(P_prox, self.xstrain)
            self.applied_forces[VP] += Fext
            self.applied_forces[WP] += Mext
            self.applied_stress[:]  += Sext

    # Declare some standard custom outputs
    def output_deflections(self, stations=(-1,)):
        def f(system):
            X = self.modes.X(self.xstrain)
            X[:,0] -= self.modes.x
            return X[stations]
        return CustomOutput(f, label="{} deflections".format(self.name))

    def output_rotations(self, stations=(-1,)):
        def f(system):
            return self.modes.small_rots(self.xstrain)[stations]
        return CustomOutput(f, label="{} rotations".format(self.name))

    def output_positions(self, stations=(-1,)):
        def f(system):
            return self.station_positions()[stations]
        return CustomOutput(f, label="{} positions".format(self.name))


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

        # basic mode shapes
        U1 = self.modes.shapes
        # extra axial motion due to transverse deflection
        U2 = np.zeros_like(U1)
        for i in range(U1.shape[0]):
            yslope =  dot(self.modes.rotations[i,2,:], self.xstrain)
            zslope = -dot(self.modes.rotations[i,1,:], self.xstrain)
            U2[i,0,:] = (yslope*self.modes.rotations[i,2,:]-
                         zslope*self.modes.rotations[i,1,:])

        # Rp * shapes
        RU = np.einsum('ij,hjp', self.Rp, (U1 - U2))

        # extra

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
