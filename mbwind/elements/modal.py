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
        if reference_spin_speed is None:
            reference_spin_speed = 1.0
        axial_force_scale = reference_spin_speed**2
        w, shapes = fe.normal_modes(num_modes, axial_force_scale)
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
                       Om[0] * S2[0, 1])),               # XXX was Om[1]?
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
        Qcf_R = -dot3(skewmat(Om), Ys, Om)
        Qco_R = 2 * dot(skewmat(Om), dot(self.S1, self.vstrain))

        Qcf_w = dot3(skewmat(Om), J, Om)
        Qco_w = 2 * dot(I_Wf2, self.vstrain)

        Qcf_f = -dot(I_Wf2.T, Om)
        Qco_f = -2 * np.einsum('i, ipq, q -> p', Om, BSB_hat, self.vstrain)

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
        local_wp_sq = np.sum(dot(self.Rp.T, self.vp[WP])[1:]**2)
        Ks = self.Ks * local_wp_sq
        if self.reference_spin_speed > 0:
            Ks /= self.reference_spin_speed**2

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
        if reference_spin_speed is None:
            reference_spin_speed = 1.0
        B = fe.Bdof & fe.Bbound
        left_boundary_dofs = sum(B[:6])
        right_boundary_dofs = sum(B[-6:])
        assert left_boundary_dofs > 0 and right_boundary_dofs > 0, \
            "Expected clamped-clamped beam"

        # Get mode shapes
        axial_force_scale = reference_spin_speed**2
        w, Phi = fe.normal_modes(num_modes, axial_force_scale)
        Xi = fe.attachment_modes(axial_force_scale)

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
