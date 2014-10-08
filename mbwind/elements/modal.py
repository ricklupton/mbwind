"""
Modal elements
"""

import numpy as np
from numpy import array, zeros, eye, dot, pi, newaxis
from scipy import linalg
from ..integrator import CustomOutput
from ..base_element import Element
from ..utils import eps_ijk, skewmat
from . import _modal_calcs

# Slices to refer to parts of matrices
VP = slice(0, 3)
WP = slice(3, 6)
VD = slice(6, 9)
WD = slice(9, 12)


def dot3(x, A, y):
    return dot(x, dot(A, y))


def inertia_matrix(S2, qn):
    J = zeros((3, 3))
    J[0, 0] = dot3(qn.T, S2[1, 1] + S2[2, 2], qn)
    J[1, 1] = dot3(qn.T, S2[2, 2] + S2[0, 0], qn)
    J[2, 2] = dot3(qn.T, S2[0, 0] + S2[1, 1], qn)
    J[0, 1] = -dot3(qn.T, S2[1, 0], qn)
    J[0, 2] = -dot3(qn.T, S2[2, 0], qn)
    J[1, 2] = -dot3(qn.T, S2[2, 1], qn)
    J[1, 0] = J[0, 1].T
    J[2, 0] = J[0, 2].T
    J[2, 1] = J[1, 2].T
    return J


class ModalElementFromFE(Element):
    _ndistal = 0
    _nstrain = 1
    _nconstraints = 0

    def __init__(self, name, fe, num_modes=None, damping=0.0,
                 reference_spin_speed=None):
        '''General element represented by finite element model.

        Geometric/centrifugal stiffening is already included in FE
        model. If the angular velocity at which the stiffening was
        calculated is provided as `reference_spin_speed`, the
        stiffening will be corrected to the current instantaneous
        angular velocity.

        '''
        # Get mode shapes
        w, shapes = fe.normal_modes(num_modes)
        self._nstrain = len(w)

        Element.__init__(self, name)
        self.loading = None
        self.fe = fe
        self.shapes = shapes
        self.mode_w = w
        self.reference_spin_speed = reference_spin_speed

        # Calculate projected matrices
        self.M = dot3(shapes.T, fe.M, shapes)
        self.K = dot3(shapes.T, fe.K, shapes)
        self.Ks = dot3(shapes.T, fe.Ks, shapes)
        self.S1 = dot(fe.S1, shapes)
        self.F1 = fe.F1
        self.S2 = np.einsum('ap, ijab, bq -> ijpq', shapes, fe.S2, shapes)
        self.F2 = np.einsum('ap, ijab     -> ijpb', shapes, fe.F2)

        # Set constant parts of mass matrix
        self.mass_vv[VP, VP] = self.fe.mass * eye(3)
        self.mass_ee[:, :] = self.M

        # Damping matrix
        self.C = np.diag(2 * damping * np.diag(self.K) / self.mode_w)
        self.C[np.isnan(self.C)] = 0.0

    def station_positions(self):
        q = self.fe.q0 + dot(self.shapes, self.xstrain)
        X = q.reshape((-1, 6))[:, :3]
        global_pos = self.rp[newaxis, :] + np.einsum('ij,hj', self.Rp, X)
        return global_pos

    def elastic_deflections(self):
        X = dot(self.shapes, self.xstrain)
        return np.c_[X[0::6], X[1::6], X[2::6]]

    def elastic_velocities(self):
        V = dot(self.shapes, self.vstrain)
        return np.c_[V[0::6], V[1::6], V[2::6]]

    def calc_mass(self):
        qn = self.fe.q0 + dot(self.shapes, self.xstrain)
        S2 = self.fe.S2
        A = self.Rp
        Ys = skewmat(dot(self.fe.S1, qn))
        Om = dot(A.T, self.vp[WP])

        # Intermediate calculations
        J = inertia_matrix(S2, qn)
        I_Wf = np.vstack((dot(qn.T, S2[1, 2] - S2[1, 2].T),
                          dot(qn.T, S2[2, 0] - S2[2, 0].T),
                          dot(qn.T, S2[0, 1] - S2[0, 1].T)))
        I_Wf2 = np.vstack((
            dot(qn.T, (Om[0] * (S2[1, 1] + S2[2, 2]) -
                       Om[1] * S2[1, 0] -
                       Om[2] * S2[2, 0])),
            dot(qn.T, (Om[1] * (S2[2, 2] + S2[0, 0]) -
                       Om[2] * S2[2, 1] -
                       Om[1] * S2[0, 1])),               # XXX should be Om[0]?
            dot(qn.T, (Om[2] * (S2[0, 0] + S2[1, 1]) -
                       Om[0] * S2[0, 2] -
                       Om[1] * S2[1, 2]))))
        I_Wf = dot(I_Wf, self.shapes)
        I_Wf2 = dot(I_Wf2, self.shapes)
        BSB_hat = array([
            self.S2[1, 2] - self.S2[1, 2].T,
            self.S2[2, 0] - self.S2[2, 0].T,
            self.S2[0, 1] - self.S2[0, 1].T,
        ])

        # Mass matrix
        M_Rw = -dot3(A, Ys, A.T)
        M_Rf = dot(A, self.S1)
        M_ww = dot3(A, J, A.T)
        M_wf = dot(A, I_Wf)

        #self.mass_vv[VP, VP] already done
        self.mass_vv[VP, WP] = M_Rw
        self.mass_vv[WP, VP] = M_Rw.T
        self.mass_vv[WP, WP] = M_ww
        self.mass_ve[VP, :] = M_Rf
        self.mass_ve[WP, :] = M_wf
        #self.mass_ee[:, :] already done

        # Velocity-dependent forces: CentriFugal and COriolis
        Qcf_R = dot3(skewmat(Om), Ys, Om)
        Qco_R = -2 * dot(skewmat(Om), dot(self.S1, self.vstrain))

        Qcf_w = -dot3(skewmat(Om), J, Om)
        Qco_w = -2 * dot(I_Wf2, self.vstrain)

        Qcf_f = dot(I_Wf2.T, Om)
        Qco_f = 2 * np.einsum('i, ipq, q -> p', Om, BSB_hat, self.vstrain)

        self.quad_forces[VP] = dot(A, (Qcf_R + Qco_R))
        self.quad_forces[WP] = dot(A, (Qcf_w + Qco_w))
        self.quad_stress[:] = Qcf_f + Qco_f

    def calc_forces_from_distributed_loading(self, load):
        F = np.zeros(self.fe.M.shape[1], dtype=load.dtype)
        for i in range(3):
            F[i::6] = load[:, i]

        # Skew torque bit
        q = self.fe.q0 + dot(self.shapes, self.xstrain)
        I = np.vstack((
            dot(q.T, (self.fe.F2[1, 2] - self.fe.F2[2, 1])),
            dot(q.T, (self.fe.F2[2, 0] - self.fe.F2[0, 2])),
            dot(q.T, (self.fe.F2[0, 1] - self.fe.F2[1, 0])),
        ))

        Qmat = np.vstack((self.fe.F1, I))
        applied_forces = dot(Qmat, F)

        # Calculate equivalent nodal forces in FE model
        Q = dot(self.fe.F, F)
        applied_stress = dot(self.shapes.T, Q)

        return applied_forces, applied_stress

    def apply_distributed_loading(self, load):
        # XXX this may not work well with +=: when is it reset?
        forces, stress = self.calc_forces_from_distributed_loading(load)
        self.applied_forces[0:3] += dot(self.Rp, forces[:3])
        self.applied_forces[3:6] += dot(self.Rp, forces[3:])
        self.applied_stress[:] += -stress  # NB sign is -ve

    def calc_external_loading(self):
        # Gravity acceleration
        acc = np.tile(self.gravity_acceleration(), 1 + self._ndistal)
        self.applied_forces[:] = dot(self.mass_vv, acc)
        self.applied_stress[:] = -dot(self.mass_ve.T, acc)
        # XXX NB applied_stresses are negative (so they are as expected for
        #     elastic stresses, but opposite for applied stresses)

        # Centrifugal stiffening: Calculate magnitude of angular
        # velocity perpendicular to beam
        if self.reference_spin_speed is not None and \
           self.reference_spin_speed > 0:
            local_wp_sq = np.sum(dot(self.Rp.T, self.vp[WP])[1:]**2)
            Ks = self.Ks * (local_wp_sq / self.reference_spin_speed**2)
        else:
            Ks = self.Ks

        # Constitutive loading
        self.applied_stress[:] += (
            dot(self.K + Ks, self.xstrain) + dot(self.C, self.vstrain))

        # External loading
        if self.loading is not None:
            if callable(self.loading):
                time = self.system.time if self.system else 0
                P_prox = self.loading(self, time)
            else:
                P_prox = np.asarray(self.loading)
            self.apply_distributed_loading(P_prox)

    # Declare some standard custom outputs
    def output_deflections(self, stations=(-1,)):
        def f(system):
            X = self.elastic_deflections()
            return X[stations]
        return CustomOutput(f, label="{} deflections".format(self.name))

    def output_positions(self, stations=(-1,)):
        def f(system):
            X = self.station_positions()
            return X[stations]
        return CustomOutput(f, label="{} positions".format(self.name))


class DistalModalElementFromFE(ModalElementFromFE):
    _ndistal = 1
    _nstrain = None  # set in __init__
    _nconstraints = 6

    def __init__(self, name, fe, num_modes=None, damping=0.0,
                 reference_spin_speed=None):
        '''General element represented by finite element model.

        Geometric/centrifugal stiffening is already included in FE
        model.

        '''
        B = fe.Bdof & fe.Bbound
        left_boundary_dofs = sum(B[:6])
        right_boundary_dofs = sum(B[-6:])
        assert left_boundary_dofs > 0 and right_boundary_dofs > 0, \
            "Expected clamped-clamped beam"

        # Get mode shapes
        w, Phi = fe.normal_modes(num_modes)
        Xi = fe.attachment_modes()

        # Discard proximal node's attachment modes
        Xi = Xi[:, left_boundary_dofs:]
        self.shapes = shapes = np.c_[Xi, Phi]

        self._nstrain = num_modes + Xi.shape[1]
        Element.__init__(self, name)
        self.loading = None
        self.fe = fe
        self.reference_spin_speed = reference_spin_speed

        # Calculate projected matrices
        self.M = dot(shapes.T, dot(fe.M, shapes))
        self.K = dot3(shapes.T, fe.K, shapes)
        self.Ks = dot3(shapes.T, fe.Ks, shapes)
        self.S1 = dot(fe.S1, shapes)
        self.F1 = fe.F1
        self.S2 = np.einsum('ap, ijab, bq -> ijpq', shapes, fe.S2, shapes)
        self.F2 = np.einsum('ap, ijab     -> ijpb', shapes, fe.F2)

        # Estimate a frequency for the attachment modes so it can be
        # used for damping calculations.
        Na = Xi.shape[1]
        with np.errstate(invalid='ignore'):
            wa = np.sqrt(np.diag(self.K)[:Na] / np.diag(self.M)[:Na])
        self.mode_w = np.r_[wa, w]

        # Set constant parts of mass matrix
        self.mass_vv[VP, VP] = self.fe.mass * eye(3)
        self.mass_ee[:, :] = self.M

        # Stiffness matrix
        # self.stiffness = np.diag(self.K)
        self.C = np.diag(2 * damping * np.diag(self.K) / self.mode_w)
        self.C[np.isnan(self.C)] = 0.0

    def calc_distal_pos(self):
        qn = self.fe.q0 + dot(self.shapes, self.xstrain)
        self.rd[:] = self.rp + dot(self.Rp, qn[-6:-3])
        Rx = np.eye(3) + skewmat(qn[-3:])
        self.Rd[:, :] = dot(self.Rp, Rx)

    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvv * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """
        𝛷d = self.shapes[-6:]
        Xd = self.fe.q0[-6:] + dot(𝛷d, self.xstrain)
        Vd = dot(𝛷d, self.vstrain)

        # Proximal nodal terms
        self.F_vp[0:3, 3:6] = -skewmat(dot(self.Rp, Xd[0:3]))

        # Strain terms
        self.F_ve[0:3, :] = dot(self.Rp, 𝛷d[0:3])
        self.F_ve[3:6, :] = dot(self.Rp, 𝛷d[3:6])

        # Quadratic accelerations:
        # 2 𝛚̃_p 𝐑_p Ẋ_d + 𝛚̃_p 𝛚̃_p 𝐑_p X_d
        self.F_v2[0:3] = dot(self.wps, (2 * dot(self.Rp, Vd[0:3]) +
                                        dot(self.wps, dot(self.Rp, Xd[0:3]))))

        # 𝛚̃_p 𝐑_p 𝛉̇_d
        self.F_v2[3:6] = dot(self.wps, dot(self.Rp, Vd[3:6]))


class DistalModalElementFromScratch(Element):
    _ndistal = 1
    _nstrain = 6
    _nconstraints = 6

    def __init__(self, name, modes, num_normal_modes=0, loading=None):
        '''
        General element represented by mode shapes. Assume no distal nodes.

         - modes   : ModalRepresentation instance describing the
                     modal representation
         - loading : Loading object describing external loading
        '''
        self._nstrain = 6 + num_normal_modes
        Element.__init__(self, name)
        self.modes = modes
        self.loading = loading

        # Reduced mass matrix
        Xi = modes.attachment_modes()
        self.mode_freqs, Phi = modes.normal_modes(clamped=True, n_modes=num_normal_modes)
        self.CB_reduction = CB = np.c_[Xi, Phi]
        self.M = dot(CB.T, dot(modes.M, CB))
        self.K = dot(CB.T, dot(modes.K, CB))

        self.damping = 0.01

    def local_deflections(self):
        """Vector deflection in proximal coordinates at each point"""
        X0 = np.zeros((len(self.modes.x), 3))
        X0[:, 0] = self.modes.x
        y = dot(self.CB_reduction[:, 6:], self.xstrain)
        defl = np.c_[y[0::6], y[1::6], y[2::6]]
        return X0, defl

    def calc_distal_pos(self):
        # Attachment modal amplitudes give relative distal position directly
        X = array([self.modes.x[-1], 0, 0]) + self.xstrain[0:3]
        R = eye(3) + skewmat(self.xstrain[3:6])
        self.rd[:] = self.rp + dot(self.Rp, X)
        self.Rd[:, :] = dot(self.Rp, R)

    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [ad w'd] = Fvp * [ap w'p]  +  Fve * [astrain]  +  Fv2
        """

        # Distal relative position in local & global coordinates
        X = array([self.modes.x[-1], 0, 0]) + self.xstrain[0:3]
        x = dot(self.Rp, X)

        # distal vel
        self.F_vp[VP, VP] = eye(3)
        self.F_vp[VP, WP] = -skewmat(x)
        self.F_vp[WP, VP] = zeros(3)
        self.F_vp[WP, WP] = eye(3)

        # Distal vel due to strain (i.e. attachment mode amplitudes)
        self.F_ve[0:3, 0:3] = self.Rp
        self.F_ve[3:6, 3:6] = self.Rp
        self.F_ve[:, 6:] = 0  # no distal vel due to normal modes

        # Quadratic velocity terms
        self.F_v2[VP] = dot(self.wps, (dot(self.wps, x) +
                                       2 * dot(self.Rp, self.vstrain[0:3])))
        self.F_v2[WP] = dot(self.wps, dot(self.Rp, self.vstrain[3:6]))

    def calc_mass(self):
        # There are lots of things that should be included to do this properly.
        # But for now, assuming the rotations of the platform are not too
        # great, I'm going to ignore them all and just transform the reduced
        # mass matrix based on the proximal orientation.

        P = np.eye(6 + self._nstrain)
        P[0:3, 0:3] = P[3:6, 3:6] = P[6:9, 6:9] = P[9:12, 9:12] = self.Rp.T
        M = dot(P.T, dot(self.M, P))
        self.mass_vv[:, :] = M[0:12, 0:12]  # proximal and distal nodes
        self.mass_ve[:, 6:] = M[0:12, 12:]  # normal modes -- node coupling
        self.mass_ee[6:, 6:] = M[12:, 12:]    # normal mode mass

        # absolute velocity of nodes, skew
        V = np.zeros((6 + self._nstrain, 3))
        V[VP, :] = skewmat(self.vp[VP])
        V[VD, :] = skewmat(self.vd[VP])
        term1 = dot(M, dot(V, self.vp[WP]))

        W = zeros((6 + self._nstrain, 6 + self._nstrain))
        W[VP, VP] = W[VD, VD] = skewmat(self.vp[WP])
        W[WP, WP] = skewmat(self.vp[WP])
        W[WD, WD] = skewmat(self.vd[WP])
        qdot = np.r_[self.vp[:], self.vd[:], self.vstrain[6:]]
        term2 = dot(W, dot(M, qdot))

        U = zeros((3, 6 + self._nstrain))
        U[:, WP] = eye(3)
        term3 = dot(dot(U.T, V.T), dot(M, qdot))

        self.quad_forces[:] = (term1 + term2 + term3)[:12]
        self.quad_stress[6:] = (term1 + term2 + term3)[12:]

    def calc_external_loading(self):
        # Gravity acceleration
        acc = np.tile(self.gravity_acceleration(), 1 + self._ndistal)
        self.applied_forces[:] = dot(self.mass_vv, acc)
        self.applied_stress[:] = -dot(self.mass_ve.T, acc)
        # XXX NB applied_stresses are negative (so they are as expected for
        #     elastic stresses, but opposite for applied stresses)

        # Constitutive loading

        # Stiffness matrix from external point of view in current configuration
        # Y = eye(15)
        # Y[:12, :12] = self.modes.transformation_to_global_coords(
        #     self.Rp, self.rp, self.rd)
        # Kext = dot(Y.T, dot(self.K, Y))

        # F = dot(Kext, self.xstrain)
        F = dot(self.K, np.r_[zeros(6), self.xstrain])

        # These forces are in local coordinates
        self.applied_forces[VP] += -dot(self.Rp, F[VP])
        self.applied_forces[WP] += -dot(self.Rp, F[WP])
        self.applied_forces[VD] += -dot(self.Rp, F[VD])
        self.applied_forces[WD] += -dot(self.Rp, F[WD])
        self.applied_stress[6:] += F[12:]

        # Damping
        self.applied_stress[:] += self.damping * self.vstrain[:]

        # # Geometric stiffness
        # if OPT_GEOMETRIC_STIFFNESS:
        #     # Calculate magnitude of angular velocity perpendicular to beam
        #     local_wp_sq = np.sum(dot(self.Rp.T, self.vp[WP])[1:]**2)
        #     self.applied_stress[:] += local_wp_sq * dot(self.kG, self.xstrain)

        # # External loading
        # if self.loading is not None:
        #     # Positions and orientations of all stations
        #     P_prox = self._get_loading()
        #     Fext,Mext,Sext = self.modes.distributed_loading(P_prox, self.xstrain)
        #     self.applied_forces[VP] += Fext
        #     self.applied_forces[WP] += Mext
        #     self.applied_stress[:]  += Sext


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

        self.stiffness = np.nan_to_num(self.modes.stiffness)
        self.damping = (2 * self.stiffness / damping_freqs *
                        np.nan_to_num(self.modes.damping[None, :]))
        #self.damping[np.isnan(self.damping)] = 0.0

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
        acc = np.tile(self.gravity_acceleration(), 1 + self._ndistal)
        self.applied_forces[:] = dot(self.mass_vv, acc)
        self.applied_stress[:] = -dot(self.mass_ve.T, acc)
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
