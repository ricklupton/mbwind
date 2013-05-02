
from __future__ import division
import numpy as np
from numpy import array, zeros, eye, newaxis, dot, r_, ix_
from scipy.integrate import trapz
from scipy import linalg
from .utils import eps_ijk, discont_trapz, skewmat
from .core import qrot3


def e_ikl_S_kl(S):
    return array([
        S[1, 2] - S[2, 1],
        S[2, 0] - S[0, 2],
        S[0, 1] - S[1, 0],
    ])


def integrate_mass_moments(X0, density):
    """
    Calculate 1st and 2nd moments of mass using exact formulae
    (trapz or simps not exact because of x or x**2 in integrand)

    Parameters
    ----------

    X0: array shape (N_stn, 3)
        location of centres of cross-sections

    density: array shape (N_stn - 1, 2)
        density at start and end of each segment
    """

    # XXX neglecting non-straight mass axis
    #     and section inertia

    m = 0
    I = zeros((3,))
    J = zeros((3, 3))
    for i in range(len(X0) - 1):
        rho1, rho2 = density[i]
        x1, y1, z1 = X0[i]
        x2, y2, z2 = X0[i+1]

        Ix = ((2*rho2+rho1)*x2**3 + (2*rho1+rho2)*x1**3 -
              3*rho1*x2*x1**2 - 3*rho2*x1*x2**2) / (6*(x2-x1))
        Iy = (rho1*y1 + rho2*y2 - (rho1*y2+rho2*y1)/2) * (x2-x1)/3
        Iz = (rho1*z1 + rho2*z2 - (rho1*z2+rho2*z1)/2) * (x2-x1)/3

        m += (x2 - x1) * (rho1 + rho2) / 2
        I += [Ix, Iy, Iz]
        J[0, 0] += ((rho1+3*rho2)*x2**4 + (rho2+3*rho1)*x1**4 -
                    4*x1*x2*(rho1*x1**2 + rho2*x2**2)) / (12*(x2-x1))

    return m, I, J


class ModalRepresentation(object):
    r"""
    Modal representation of a 1-dimensional element.

    Mode shapes
    ===========

    The mode shapes are defined by the matrix :math:`\boldsymbol{U}`, with one
    column for each mode shape, defined at a set of points correspondng to the
    rows.

    Shape integrals
    ===============

    The equations of motion using the modal representation include integrals
    over the deformed mass distribution, which is

    .. math::

        \boldsymbol{X}(\boldsymbol{X}_0) =
            \boldsymbol{X}_0 +
            \boldsymbol{U}(\boldsymbol{X}_0) \boldsymbol{\epsilon}

    where :math:`\boldsymbol{X}_0` is an undeflected position where the density
    and modeshapes are defined.

    Firstly, the mass, moment of mass and inertia tensor are required. These
    are integrals over the undeflected mass distribution only:

    .. math::

        m &= \int dm

        \boldsymbol{I}_0 &= \int \boldsymbol{X}_0 dm

        \boldsymbol{J}
            &= - \int \boldsymbol{\tilde{X}}_0 \boldsymbol{\tilde{X}}_0 dm

    The velocity and acceleration due to changes in modal amplitude depend only
    on the mode shapes, and thus require the first shape integral,

    .. math::

        \boldsymbol{S} &= \int \boldsymbol{U} dm

    The rotational inertia depends on the deflected position squared, which
    gives products requiring the second and third shape integrals,

    .. math::

        \boldsymbol{S}_j
            &= - \int \boldsymbol{\tilde{X}}_0 \boldsymbol{\tilde{U}}_j dm

        \boldsymbol{S}_{jk}
            &= - \int \boldsymbol{\tilde{U}}_j \boldsymbol{\tilde{U}}_k dm

    Two more integrals are defined:

    .. math::

        \boldsymbol{T}
            &= - \int \boldsymbol{\tilde{X}}_0 \boldsymbol{U} dm

        \boldsymbol{T}_j
            &= - \int \boldsymbol{\tilde{U}}_j \boldsymbol{U} dm

    Strictly fewer integrals are needed, see for example "Dynamics of Multibody
    Systems" ch. 5 by A. Shabana.

    These integrals are evaluated using the trapezium rule using the provided
    density distribution and mode shapes. Because the element is 1-dimensional,
    the inertia of the cross-sections is not included. An extra inertia
    distribution can be provided which is added to the calculation of the
    inertia tensor.

    This probably neglects the change in this added inertia with deflection,
    but that is probably ok.

    Because the element is one-dimensional, the above formulae can be written
    explicitly as

    .. math::

        \boldsymbol{X}(s) = \begin{bmatrix} s \\ 0 \\ 0 \end{bmatrix}
            + \boldsymbol{U}(\boldsymbol{X}_0) \boldsymbol{\epsilon}

        I_0 &= \int s \rho(s) \, ds

        \boldsymbol{J}
            &= \begin{bmatrix} 0 & 0 & 0 \\ 0 & J_0 & 0 \\ 0 & 0 & J_0
               \end{bmatrix}
            \mbox{ where } J_0 &= \int s^2  \rho(s) \, ds

        \boldsymbol{S}_i &= \int \boldsymbol{U}_i(s) \rho(s) \, ds

        \boldsymbol{T}_i &= \int \begin{bmatrix}
            0 & 0 & 0 \\ 0 & 0 & s \\ 0 & -s & 0 \end{bmatrix}
            \boldsymbol{\tilde{U}}_i(s) \rho(s) \, ds

        \boldsymbol{\tilde{S}}_{ij}
            &= - \int \boldsymbol{\tilde{U}}_i(s) \boldsymbol{\tilde{U}}_j(s)
            \rho(s) \, ds

    """
    def __init__(self, x, shapes, rotations, density, freqs, mass_axis=None,
                 section_inertia=None, gyration_ratio=1.0,
                 damping=None, mode_names=None):
        r"""
        Setup modal representation for a 1-dimensional element.

        Parameters
        ----------
        x : array (`N_x`)
            Coordinates where mode shapes and densities are defined

        shapes : array (`N_x` x 3 x `N_modes`)
            Array of mode shapes defined at points in `x`

        rotations : array (`N_x` x 3 x `N_modes`)
            Array of mode rotations defined at points in `x`

        density : array (`N_x`)
            Linear density distribution at points defined in `x`

        mass_axis : array(`N_x` x 2)
            Y- and Z- coordinates of centre of mass at each point

        section_inertia : array (`N_x`)
            Polar inertia of cross-sections

        gyration_ratio : array (`N_x`) or None
            Ratio of radius of gyration about y-axis to that about the z-axis.
            Assumes cross-sections are laminae so the transverse inertias add
            to give the required polar inertia.

            If None, transverse inertias are zero.

        freqs : array (`N_modes`)
            Array of modal frequencies

        damping : array (`N_modes`), default zeros
            Array of modal damping factors

        """
        assert len(x) == shapes.shape[0] == rotations.shape[0]
        assert len(freqs) == shapes.shape[2] == rotations.shape[2]
        assert shapes.shape[1] == rotations.shape[1] == 3

        if damping is None:
            damping = zeros(len(freqs))

        if mass_axis is None:
            mass_axis = zeros((len(x), 2))

        self.section_inertia = zeros((len(x), 3, 3))
        Ixx = Iyy = Izz = 0
        if section_inertia is not None:
            Ixx = section_inertia  # Polar inertia
            if gyration_ratio is not None:
                # Transverse inertia: gyration_ratio = Iyy / Izz
                Izz = Ixx / (1 + gyration_ratio)
                Iyy = Ixx - Izz                  # parallel axis theorem
        self.section_inertia[:, 0, 0] = Ixx
        self.section_inertia[:, 1, 1] = Iyy
        self.section_inertia[:, 2, 2] = Izz

        # If only one set of density values is given, assume it's continuous.
        # Otherwise, values should be given at start and end of each section.
        if density.ndim == 1:
            self.density = zeros((len(x)-1, 2))
            self.density[:, 0] = density[:-1]
            self.density[:, 1] = density[1:]
        elif density.ndim == 2:
            self.density = density
        else:
            raise ValueError('density should be Nx x 1 or Nx x 2')

        self.x = x
        self.shapes = shapes
        self.rotations = rotations
        self.freqs = freqs
        self.damping = damping
        self.mass_axis = mass_axis

        # describe mode shapes
        if mode_names is None:
            mode_names = ['Mode %d' % i for i in range(len(freqs))]
        self.mode_names = mode_names

        # location of centre of each cross-section
        self.X0 = X0 = np.c_[x, mass_axis]

        # Integrate mass (un-deflected shape integrals). This is done
        # exactly if the density distribution is linear.
        self.mass, self.I0, self.J0 = integrate_mass_moments(X0, self.density)

        # Prepare integrands
        Nfreq = len(self.freqs)
        J0 = zeros((len(x), 3, 3))
        S1 = zeros((len(x), 3, 3, Nfreq))
        T1 = zeros((len(x), 3, 3, Nfreq))
        S2 = zeros((len(x), 3, 3, Nfreq, Nfreq))
        T2 = zeros((len(x), 3, 3, Nfreq, Nfreq))
        # H2 = zeros((len(x), 3, Nfreq, Nfreq))
        # H3 = zeros((len(x), 3, Nfreq, Nfreq, Nfreq))

        for i in range(len(x)):
            J0[i] = np.outer(X0[i], X0[i]) + self.section_inertia[i]

            S1[i] = np.einsum('j,ip', X0[i], self.shapes[i])
            S2[i] = np.einsum('ip,jr', self.shapes[i], self.shapes[i])

            T1[i] = np.einsum('inq,nj,qp', eps_ijk,
                              self.section_inertia[i], self.rotations[i])
            T2[i] = np.einsum('ist,jlm,ls,mp,tr', eps_ijk, eps_ijk,
                              self.section_inertia[i],
                              self.rotations[i], self.rotations[i])

            # H2[i] = (np.einsum('p,q,j->jpq', self.rotations[i, 1],
            #                    self.rotations[i, 1], X0[i]) +
            #          np.einsum('p,q,j->jpq', self.rotations[i, 2],
            #                    self.rotations[i, 2], X0[i]))
            # H3[i] = (np.einsum('p,q,jr->jpqr', self.rotations[i, 1],
            #                    self.rotations[i, 1], self.shapes[i]) +
            #          np.einsum('p,q,jr->jpqr', self.rotations[i, 2],
            #                    self.rotations[i, 2], self.shapes[i]))

        # Calculate shape integrals, using trapezium rule (assume
        # linear variation across segments) and allowing that density
        # may contain discontinuities.
        self.S = discont_trapz(self.shapes, self.density, x)
        self.S1 = discont_trapz(S1, self.density, x)
        self.T1 = discont_trapz(T1, self.density, x)
        self.S2 = discont_trapz(S2, self.density, x)
        self.T2 = discont_trapz(T2, self.density, x)

        self.S1b = self.S1 + self.T1
        self.S2b = self.S2 + self.T2

        # Might as well finish this change now - always done in inertia_tensor
        self.ss_J0 = (eye(3)[:, :] * self.J0.trace()) - self.J0
        self.ss_S1 = (eye(3)[:, :, newaxis] * self.S1b.trace()) - self.S1b
        self.ss_S2 = ((eye(3)[:, :, newaxis, newaxis] * self.S2b.trace())
                      - self.S2b)
        self.e_S1 = e_ikl_S_kl(self.S1b)
        self.e_S2 = e_ikl_S_kl(self.S2b)

        # Add in section inertias
        self.ss_J0 += trapz(self.section_inertia, x, axis=0)

        # Skew forms of I0 and S
        self.sk_I0 = np.einsum('imj,m ->ij ', eps_ijk, self.I0)
        self.sk_S0 = np.einsum('imj,mp->ijp', eps_ijk, self.S)

    def X(self, q):
        return self.X0 + np.einsum('hip,p', self.shapes, q)

    def Xdot(self, qd):
        return np.einsum('hip,p', self.shapes, qd)

    def small_rots(self, q):
        return np.einsum('hip,p', self.rotations, q)

    def R(self, q):
        rotations = np.einsum('hip,p', self.rotations, q)
        R = zeros((len(self.x), 3, 3))
        for i in range(len(self.x)):
            R[i] = qrot3(rotations[i]/2)
        return R

    def inertia_tensor(self, q):
        """
        Construct the inertia tensor corresponding to the modal coordinates

        .. math::

            I_{\theta\theta} = (\delta_{ij}J0_{kk} - J0_{ij})
                + [ \delta_{ij}(2 S_{kkp}) - (S_{ijp} + S_{jip}) +
                    (T_{ijp} + T_{jip}) ] \epsilon_p
                + [ \delta_{ij}(S_{kkpr} + T_{kkpr}) -
                    (S_ijpr + T_ijpr) ] \epsilon_p \epsilon_r

        """
        #inertia = eye(3)*self.J0.trace() - self.J0
        if len(self.freqs) > 0:
            S1 = np.einsum('ijp,p', self.S1, q)
            T1 = np.einsum('ijp,p', self.T1, q)
            S2 = np.einsum('ijpr,p,r', self.S2, q, q)
            T2 = np.einsum('ijpr,p,r', self.T2, q, q)
            inertia = (self.ss_J0 + eye(3)*2*S1.trace() -
                       (S1 + S1.T) - (T1 + T1.T) +
                       eye(3)*(S2.trace() + T2.trace()) - (S2 + T2))
        else:
            inertia = self.ss_J0.copy()
        return inertia

    def strain_strain(self):
        if len(self.freqs) > 0:
            A = np.einsum('mmpr', self.S2) + np.einsum('mmpr', self.T2)
        else:
            A = np.zeros((0, 0))
        return A

    def rotation_strain(self, q):
        if len(self.freqs) > 0:
            A = (self.S1 + self.T1 + np.einsum('klpr,r', self.S2, q) +
                 np.einsum('klrp,r', self.T2, q))
            B = np.einsum('ikl,lkp', eps_ijk, A)
        else:
            B = np.zeros((3, 0))
        return B

    def inertia_vel(self, q, qd):
        if len(self.freqs) > 0:
            S1 = np.einsum('ijp,p', self.S1, qd)
            T1 = np.einsum('ijp,p', self.T1, qd)
            S2 = np.einsum('ijpr,p,r', self.S2, qd, q)
            T2 = np.einsum('ijpr,p,r', self.T2, qd, q)
            A = (eye(3)*(S1 + T1).trace() - S1 - T1 +
                 eye(3)*(S2 + T2).trace() - S2 - T2)
        else:
            A = zeros((3, 3))
        return A

    def quad_stress(self, q, qd, Wp):
        if len(self.freqs) > 0:
            A1 = self.S1 + self.T1
            A2 = (np.einsum('ijpr,r', self.S2, q) +
                  np.einsum('ijrp,r', self.T2, q))

            # depending on angular velocity
            C = (np.einsum('ijp,i,j', A1+A2, Wp, Wp) -
                 dot(Wp, Wp) * ((A1+A2).trace()))

            # depending on strain velocity
            B2 = self.S2 - self.T2.swapaxes(2, 3)
            D = 2 * np.einsum('ijk,j,l,ikpl', eps_ijk, Wp, qd, B2)

            return C + D
        else:
            return np.zeros(0)

    def distributed_loading(self, P, q):
        """
        Return the generalised forces corresponding to the distributed load P.

        Returns a tuple (Q_r, Q_w, Q_e) of forces/moments/stresses.
        """
        X = self.X(q)  # length x 3
        XcrossP = np.einsum('ijk,hj,hk->hi', eps_ijk, X, P)  # length x 3
        UTP = np.einsum('hip,hi->hp', self.shapes, P)
        Qr = -trapz(P,       self.x, axis=0)
        Qw = trapz(XcrossP, self.x, axis=0)
        Qe = -trapz(UTP,     self.x, axis=0)
        return Qr, Qw, Qe

    def geometric_stiffness(self, N):
        """
        Calculate the geometric stiffening matrix corresponding to an applied
        force N.
        """
        # loop through SEGMENTS
        #Nmode = len(self.freqs)
        #Nstns = len(self.x)
        #kG = zeros((Nmode,Nmode))
        #for i in range(Nstns-1):
        #    dx = self.x[i+1] - self.x[i]
        #    for p in range(Nmode):
        #        slope_p = self.shapes[i+1,:,p] - self.shapes[i,:,p]
        #        for q in range(Nmode):
        #            slope_q = self.shapes[i+1,:,q] - self.shapes[i,:,q]
        #            intN = trapz(N[i:i+2], self.x[i:i+2], axis=0)
        #            kG[p,q] += dot(slope_p,slope_q)/dx**2 * intN

        # XXX Assume force acts axially along the beam
        dx = np.diff(self.x)
        slopes = np.r_[
            #zeros((1,)+self.shapes.shape[1:]),
            np.diff(self.shapes, axis=0) / dx[:, newaxis, newaxis],
            zeros((1,)+self.shapes.shape[1:]),
        ]  # use slope of preceding segment
        inty = np.einsum('hp,hq->hpq', slopes[:, 1, :], slopes[:, 1, :])
        intz = np.einsum('hp,hq->hpq', slopes[:, 2, :], slopes[:, 2, :])
        kG = trapz((inty + intz) * N[:, newaxis, newaxis], x=self.x, axis=0)
        return kG

    def save(self, filename):
        np.savez(filename, x=self.x, shapes=self.shapes, freqs=self.freqs,
                 rotations=self.rotations, density=self.density)

    @classmethod
    def from_file(cls, filename):
        if not filename.lower().endswith('.npz'):
            filename += '.npz'
        npz = np.load(filename)
        return cls(npz['x'], npz['shapes'], npz['rotations'],
                   npz['freqs'], npz['density'])


class BeamModalAnalysis(object):
    '''
    Assembles several beams to calculate orthogonal mode shapes and frequencies
    '''
    def __init__(self, length, density, EA, EIy, EIz, GIx, Jx,
                 Nmodes=None, Nelements=10):

        x0 = np.linspace(0.0, length, num=Nelements+1)
        beams = []
        for i in range(Nelements):
            newbeam = UniformBeam('beam%d'%(i+1), length/Nelements,
                                density, EA, EIy, EIz, GIx, Jx)
            if beams: beams[-1].add_leaf(newbeam)
            beams.append(newbeam)
        self.system = System(beams[0])

        # Find mode shapes in terms of beam strains
        self.linsys = LinearisedSystem(self.system)
        w,v = scipy.linalg.eig(self.linsys.K, self.linsys.M)
        order = np.argsort(w)
        w = np.real(w[order])
        v = v[:,order]

        if Nmodes is None:
            Nmodes = len(w)
        assert Nmodes <= 6*Nelements

        # Find actual shapes that correspond to beam strain mode shapes
        shapes    = np.zeros((Nelements+1, 3, Nmodes))
        rotations = np.zeros((Nelements+1, 3, Nmodes))
        iDOF = self.system.iFreeDOF
        assert v.shape[0] == np.count_nonzero(iDOF)
        for imode in range(Nmodes):
            self.system.q[iDOF] = v[:,imode] / 10
            self.system.update(0.0, False)
            for ib,beam in enumerate(beams):
                qd = rotation_matrix_to_quaternions(beam.Rd)
                shapes[1+ib,:,imode] = beam.rd - [x0[1+ib],0,0]
                rotations[1+ib,:,imode] = qd[1:]

        self.rep = ModalRepresentation(x0, shapes, rotations, w[:Nmodes],
                                       np.ones_like(x0)*density)


class BladeModalAnalysis(object):
    def __init__(self, x, density, EIy, EIz):
        self.x = x
        self.density = density
        self.EIy = EIy
        self.EIz = EIz

        # Discrete mass matrix at all points
        self.M = np.zeros((len(x), len(x)))
        for i in range(0, len(x)-1):
            m1, m2 = self.density[i, :]
            x1, x2 = x[i:i+2]

            self.M[i:i+2, i:i+2] += array([
                [3*m1 + m2, m1 + m2],
                [m1 + m2,   m1 + 3*m2],
            ]) * (x2 - x1) / 12

        ########### STIFFNESS #################
        # Assume EI varies linearly in each segment and curvature is constant.
        #   V_{seg} = 0.5 \int_{seg} EI(x)
        #                   [(\theta_2 - \theta_1) / \Delta{}x]^2 dx
        # so
        #   K_{seg} = [[1, -1], [-1, 1]] * (EI1 + EI2) / (2 dx)
        self.K = zeros((len(x), len(x), 3, 3))
        for i in range(self.K.shape[0]-1):
            # XXX ignore torsion for now
            ky = EIy[i].mean() / (x[i+1] - x[i])
            kz = EIz[i].mean() / (x[i+1] - x[i])
            self.K[i:i+2, i:i+2, 1, 1] += ky * array([[1, -1], [-1, 1]])
            self.K[i:i+2, i:i+2, 2, 2] += kz * array([[1, -1], [-1, 1]])

    def modal_stiffness(self, rotations):
        # Sum over mode shapes to give modal stiffness
        return np.einsum('xip,xyij,yjq->pq', rotations, self.K, rotations)


class ModesFromScratch(object):
    def __init__(self, x, density, EA, EIy, EIz):
        assert x[0] == 0
        self.x = x
        self.density = self._prepare_inputs(len(x), density)
        self.EA = self._prepare_inputs(len(x), EA)
        self.EIy = self._prepare_inputs(len(x), EIy)
        self.EIz = self._prepare_inputs(len(x), EIz)
        if not np.allclose(self.EIy, self.EIz):
            raise NotImplementedError("Non-symmetric EI not done yet")

        # Assemble matrices
        N_nodes = len(x)
        N_dof = N_nodes * 6
        self.M = np.zeros((N_dof, N_dof))
        self.K = np.zeros((N_dof, N_dof))
        for i_el in range(N_nodes - 1):
            elem_length = x[i_el+1] - x[i_el]
            ke = self.element_stiffness_matrix(
                self.EA[i_el, 0], self.EA[i_el, 1],
                self.EIy[i_el, 0], self.EIy[i_el, 1],
                elem_length
            )
            me = self.element_mass_matrix(
                self.density[i_el, 0], self.density[i_el, 1],
                elem_length
            )
            i1, i2 = i_el * 6, (i_el + 2) * 6
            self.M[i1:i2, i1:i2] += me
            self.K[i1:i2, i1:i2] += ke

    def _prepare_inputs(self, n, input):
        # If only one set of density values is given, assume it's continuous.
        # Otherwise, values should be given at start and end of each section.
        input = np.asarray(input)
        if input.ndim == 0:
            return input * np.ones((n - 1, 2))
        elif input.ndim == 1:
            assert input.shape[0] == n
            yy = np.zeros((n - 1, 2))
            yy[:, 0] = input[:-1]
            yy[:, 1] = input[1:]
            return yy
        elif input.ndim == 2:
            assert input.shape[0] == n - 1
            return input
        else:
            raise ValueError('input should be Nx x 1 or Nx x 2')

    def element_mass_matrix(self, rho_1, rho_2, l):
        """
        Finite element mass matrix, assuming cubic shape functions and
        linear density variation from ``r1`` to ``r2``.
        """
        e = array([[l*(rho_1/4 + rho_2/12), 0, 0, 0, 0, 0, l*(rho_1/12 + rho_2/12), 0, 0, 0, 0, 0], [0, l*(2*rho_1/7 + 3*rho_2/35), 0, 0, 0, l**2*(15*rho_1 + 7*rho_2)/420, 0, l*(9*rho_1/140 + 9*rho_2/140), 0, 0, 0, l**2*(-7*rho_1 - 6*rho_2)/420], [0, 0, l*(2*rho_1/7 + 3*rho_2/35), 0, l**2*(-15*rho_1 - 7*rho_2)/420, 0, 0, 0, l*(9*rho_1/140 + 9*rho_2/140), 0, l**2*(7*rho_1 + 6*rho_2)/420, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, l**2*(-15*rho_1 - 7*rho_2)/420, 0, l**3*(rho_1/168 + rho_2/280), 0, 0, 0, l**2*(-6*rho_1 - 7*rho_2)/420, 0, l**3*(-rho_1 - rho_2)/280, 0], [0, l**2*(15*rho_1 + 7*rho_2)/420, 0, 0, 0, l**3*(rho_1/168 + rho_2/280), 0, l**2*(6*rho_1 + 7*rho_2)/420, 0, 0, 0, l**3*(-rho_1 - rho_2)/280], [l*(rho_1/12 + rho_2/12), 0, 0, 0, 0, 0, l*(rho_1/12 + rho_2/4), 0, 0, 0, 0, 0], [0, l*(9*rho_1/140 + 9*rho_2/140), 0, 0, 0, l**2*(6*rho_1 + 7*rho_2)/420, 0, l*(3*rho_1/35 + 2*rho_2/7), 0, 0, 0, l**2*(-7*rho_1 - 15*rho_2)/420], [0, 0, l*(9*rho_1/140 + 9*rho_2/140), 0, l**2*(-6*rho_1 - 7*rho_2)/420, 0, 0, 0, l*(3*rho_1/35 + 2*rho_2/7), 0, l**2*(7*rho_1 + 15*rho_2)/420, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, l**2*(7*rho_1 + 6*rho_2)/420, 0, l**3*(-rho_1 - rho_2)/280, 0, 0, 0, l**2*(7*rho_1 + 15*rho_2)/420, 0, l**3*(rho_1/280 + rho_2/168), 0], [0, l**2*(-7*rho_1 - 6*rho_2)/420, 0, 0, 0, l**3*(-rho_1 - rho_2)/280, 0, l**2*(-7*rho_1 - 15*rho_2)/420, 0, 0, 0, l**3*(rho_1/280 + rho_2/168)]])
        # e = array([
        #     [r1/4 + r2/12, 0, 0, 0, 0, 0,
        #      r1/12 + r2/12, 0, 0, 0, 0, 0],
        #     [0, 2*r1/7 + 3*r2/35, 0, 0, 0, l*(15*r1 + 7*r2)/420,
        #      0, 9*r1/140 + 9*r2/140, 0, 0, 0, l*(-7*r1 - 6*r2)/420],
        #     [0, 0, 2*r1/7 + 3*r2/35, 0, l*(-15*r1 - 7*r2)/420, 0,
        #      0, 0, 9*r1/140 + 9*r2/140, 0, l*(7*r1 + 6*r2)/420, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, l*(-15*r1 - 7*r2)/420, 0, l**2*(r1/168 + r2/280), 0,
        #      0, 0, l*(-6*r1 - 7*r2)/420, 0, l**2*(-r1 - r2)/280, 0],
        #     [0, l*(15*r1 + 7*r2)/420, 0, 0, 0, l**2*(r1/168 + r2/280),
        #      0, l*(6*r1 + 7*r2)/420, 0, 0, 0, l**2*(-r1 - r2)/280],
        #     [r1/12 + r2/12, 0, 0, 0, 0, 0,
        #      r1/12 + r2/4, 0, 0, 0, 0, 0],
        #     [0, 9*r1/140 + 9*r2/140, 0, 0, 0, l*(6*r1 + 7*r2)/420,
        #      0, 3*r1/35 + 2*r2/7, 0, 0, 0, l*(-7*r1 - 15*r2)/420],
        #     [0, 0, 9*r1/140 + 9*r2/140, 0, l*(-6*r1 - 7*r2)/420, 0,
        #      0, 0, 3*r1/35 + 2*r2/7, 0, l*(7*r1 + 15*r2)/420, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, l*(7*r1 + 6*r2)/420, 0, l**2*(-r1 - r2)/280, 0,
        #      0, 0, l*(7*r1 + 15*r2)/420, 0, l**2*(r1/280 + r2/168), 0],
        #     [0, l*(-7*r1 - 6*r2)/420, 0, 0, 0, l**2*(-r1 - r2)/280, 0,
        #      l*(-7*r1 - 15*r2)/420, 0, 0, 0, l**2*(r1/280 + r2/168)]])
        return e

    def element_stiffness_matrix(self, EA_1, EA_2, EI_1, EI_2, l):
        """
        Finite element stiffness matrix, assuming cubic shape functions and
        linear stiffness variation.
        """
        e = array([[(EA_1 + EA_2)/(2*l), 0, 0, 0, 0, 0, -(EA_1 + EA_2)/(2*l), 0, 0, 0, 0, 0], [0, 6*(EI_1 + EI_2)/l**3, 0, 0, 0, 2*(2*EI_1 + EI_2)/l**2, 0, -(6*EI_1 + 6*EI_2)/l**3, 0, 0, 0, 2*(EI_1 + 2*EI_2)/l**2], [0, 0, 6*(EI_1 + EI_2)/l**3, 0, -(4*EI_1 + 2*EI_2)/l**2, 0, 0, 0, -(6*EI_1 + 6*EI_2)/l**3, 0, -(2*EI_1 + 4*EI_2)/l**2, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, -(4*EI_1 + 2*EI_2)/l**2, 0, (3*EI_1 + EI_2)/l, 0, 0, 0, 2*(2*EI_1 + EI_2)/l**2, 0, (EI_1 + EI_2)/l, 0], [0, 2*(2*EI_1 + EI_2)/l**2, 0, 0, 0, (3*EI_1 + EI_2)/l, 0, -(4*EI_1 + 2*EI_2)/l**2, 0, 0, 0, (EI_1 + EI_2)/l], [-(EA_1 + EA_2)/(2*l), 0, 0, 0, 0, 0, (EA_1 + EA_2)/(2*l), 0, 0, 0, 0, 0], [0, -(6*EI_1 + 6*EI_2)/l**3, 0, 0, 0, -(4*EI_1 + 2*EI_2)/l**2, 0, 6*(EI_1 + EI_2)/l**3, 0, 0, 0, -(2*EI_1 + 4*EI_2)/l**2], [0, 0, -(6*EI_1 + 6*EI_2)/l**3, 0, 2*(2*EI_1 + EI_2)/l**2, 0, 0, 0, 6*(EI_1 + EI_2)/l**3, 0, 2*(EI_1 + 2*EI_2)/l**2, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, -(2*EI_1 + 4*EI_2)/l**2, 0, (EI_1 + EI_2)/l, 0, 0, 0, 2*(EI_1 + 2*EI_2)/l**2, 0, (EI_1 + 3*EI_2)/l, 0], [0, 2*(EI_1 + 2*EI_2)/l**2, 0, 0, 0, (EI_1 + EI_2)/l, 0, -(2*EI_1 + 4*EI_2)/l**2, 0, 0, 0, (EI_1 + 3*EI_2)/l]])
        #e = array([[(EA_1 + EA_2)/(2*l**2), 0, 0, 0, 0, 0, -(EA_1 + EA_2)/(2*l**2), 0, 0, 0, 0, 0], [0, 6*(EI_1 + EI_2)/l**4, 0, 0, 0, 2*(2*EI_1 + EI_2)/l**3, 0, -(6*EI_1 + 6*EI_2)/l**4, 0, 0, 0, 2*(EI_1 + 2*EI_2)/l**3], [0, 0, 6*(EI_1 + EI_2)/l**4, 0, -(4*EI_1 + 2*EI_2)/l**3, 0, 0, 0, -(6*EI_1 + 6*EI_2)/l**4, 0, -(2*EI_1 + 4*EI_2)/l**3, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, -(4*EI_1 + 2*EI_2)/l**3, 0, (3*EI_1 + EI_2)/l**2, 0, 0, 0, 2*(2*EI_1 + EI_2)/l**3, 0, (EI_1 + EI_2)/l**2, 0], [0, 2*(2*EI_1 + EI_2)/l**3, 0, 0, 0, (3*EI_1 + EI_2)/l**2, 0, -(4*EI_1 + 2*EI_2)/l**3, 0, 0, 0, (EI_1 + EI_2)/l**2], [-(EA_1 + EA_2)/(2*l**2), 0, 0, 0, 0, 0, (EA_1 + EA_2)/(2*l**2), 0, 0, 0, 0, 0], [0, -(6*EI_1 + 6*EI_2)/l**4, 0, 0, 0, -(4*EI_1 + 2*EI_2)/l**3, 0, 6*(EI_1 + EI_2)/l**4, 0, 0, 0, -(2*EI_1 + 4*EI_2)/l**3], [0, 0, -(6*EI_1 + 6*EI_2)/l**4, 0, 2*(2*EI_1 + EI_2)/l**3, 0, 0, 0, 6*(EI_1 + EI_2)/l**4, 0, 2*(EI_1 + 2*EI_2)/l**3, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, -(2*EI_1 + 4*EI_2)/l**3, 0, (EI_1 + EI_2)/l**2, 0, 0, 0, 2*(EI_1 + 2*EI_2)/l**3, 0, (EI_1 + 3*EI_2)/l**2, 0], [0, 2*(EI_1 + 2*EI_2)/l**3, 0, 0, 0, (EI_1 + EI_2)/l**2, 0, -(2*EI_1 + 4*EI_2)/l**3, 0, 0, 0, (EI_1 + 3*EI_2)/l**2]])
        return e

    def normal_modes(self, n_modes=None, clamped=True, exclude_axial=True,
                     exclude_torsion=True):
        """
        Calculate the normal mode shapes and frequencies, limited to the first
        ``n_modes`` modes if required.
        """
        assert exclude_axial and exclude_torsion

        # remove axial/torsion
        idx = [i for i in range(len(self.K)) if (i % 6) not in (0, 3)]
        if clamped:
            idx = idx[4:-4]  # take away ends

        w, v = linalg.eig(self.K[ix_(idx, idx)], self.M[ix_(idx, idx)])
        order = np.argsort(w)
        w = np.sqrt(w[order].real)
        v = v[:, order]

        # put back axial/torsion as zeros
        shapes = zeros((len(self.K), v.shape[1]))
        shapes[idx, :] = v

        if n_modes is not None:
            w = w[:n_modes]
            shapes = shapes[:, :n_modes]
        return w, shapes

    def attachment_modes(self, exclude_axial=True, exclude_torsion=True):
        """
        Calculate the mode shapes with unit deflection at the ends of the beam
        """
        assert exclude_axial and exclude_torsion

        N = self.K.shape[0]

        # remove axial/torsion
        idx_ok = array([(i % 6) != 3 for i in range(N)])
        #idx_ok = np.ones(N, dtype=bool)
        idx_B = zeros(N, dtype=bool)
        idx_B[:6] = idx_B[-6:] = True
        K_II = self.K[idx_ok & ~idx_B, :][:, idx_ok & ~idx_B]
        K_IB = self.K[idx_ok & ~idx_B, :][:, idx_ok & idx_B]

        # Attachment modes
        Xi = zeros((N, 12))
        Xi[ix_(idx_ok & ~idx_B, idx_ok[:12])] = -dot(linalg.inv(K_II), K_IB)
        Xi[ix_(idx_ok & idx_B,  idx_ok[:12])] = eye(sum(idx_ok[:12]))
        return Xi

    @property
    def K_BI(self):
        idx_B = zeros(self.K.shape[0], dtype=bool)
        idx_B[:6] = idx_B[-6:] = True
        return self.K[idx_B, :][:, ~idx_B]

    @property
    def M_BI(self):
        idx_B = zeros(self.K.shape[0], dtype=bool)
        idx_B[:6] = idx_B[-6:] = True
        return self.M[idx_B, :][:, ~idx_B]

    def transformation_to_global_coords(self, Rp, rp, rd):
        """
        Returns the transformation matrix Y which maps the global velocities
        \dot{r}_p, \omega_p, \dot{r}_d, \omega_d to the local deflections
        \dot{u}_p, \Omega_p, \dot{u}_d, \Omega_d, for the instantaneous
        configuration of the body specified by Rp, rp and rd.
        """
        Y = zeros((12, 12))
        Y[6:9, 0:3] = -Rp.T
        Y[6:9, 3:6] = dot(Rp.T, skewmat(rd - rp))
        Y[6:9, 6:9] = Rp.T
        Y[9:12, 3:6] = -Rp.T
        Y[9:12, 9:12] = Rp.T
        return Y


########################################################################
## Shape integrals etc #################################################
########################################################################


class ShapeIntegrals(object):
    def __init__(self, modes):
        """modes should be an object with M and K attributes"""
        self.modes = modes

        # location of centre of each cross-section
        #self.X0 = X0 = np.c_[x, mass_axis]
        self.X0 = zeros((len(modes.x), 3))
        self.X0[:, 0] = modes.x

        # Integrate mass (un-deflected shape integrals). This is done
        # exactly if the density distribution is linear.
        self.mass, self.I0, self.J0 = integrate_mass_moments(self.X0, modes.density)

        # Prepare integrands
        Nfreq = len(self.freqs)
        J0 = zeros((len(x), 3, 3))
        S1 = zeros((len(x), 3, 3, Nfreq))
        T1 = zeros((len(x), 3, 3, Nfreq))
        S2 = zeros((len(x), 3, 3, Nfreq, Nfreq))
        T2 = zeros((len(x), 3, 3, Nfreq, Nfreq))
        # H2 = zeros((len(x), 3, Nfreq, Nfreq))
        # H3 = zeros((len(x), 3, Nfreq, Nfreq, Nfreq))

        for i in range(len(x)):
            J0[i] = np.outer(X0[i], X0[i]) + self.section_inertia[i]

            S1[i] = np.einsum('j,ip', X0[i], self.shapes[i])
            S2[i] = np.einsum('ip,jr', self.shapes[i], self.shapes[i])

            T1[i] = np.einsum('inq,nj,qp', eps_ijk,
                              self.section_inertia[i], self.rotations[i])
            T2[i] = np.einsum('ist,jlm,ls,mp,tr', eps_ijk, eps_ijk,
                              self.section_inertia[i],
                              self.rotations[i], self.rotations[i])

            # H2[i] = (np.einsum('p,q,j->jpq', self.rotations[i, 1],
            #                    self.rotations[i, 1], X0[i]) +
            #          np.einsum('p,q,j->jpq', self.rotations[i, 2],
            #                    self.rotations[i, 2], X0[i]))
            # H3[i] = (np.einsum('p,q,jr->jpqr', self.rotations[i, 1],
            #                    self.rotations[i, 1], self.shapes[i]) +
            #          np.einsum('p,q,jr->jpqr', self.rotations[i, 2],
            #                    self.rotations[i, 2], self.shapes[i]))

        # Calculate shape integrals, using trapezium rule (assume
        # linear variation across segments) and allowing that density
        # may contain discontinuities.
        self.S = discont_trapz(self.shapes, self.density, x)
        self.S1 = discont_trapz(S1, self.density, x)
        self.T1 = discont_trapz(T1, self.density, x)
        self.S2 = discont_trapz(S2, self.density, x)
        self.T2 = discont_trapz(T2, self.density, x)

        self.S1b = self.S1 + self.T1
        self.S2b = self.S2 + self.T2

        # Might as well finish this change now - always done in inertia_tensor
        self.ss_J0 = (eye(3)[:, :] * self.J0.trace()) - self.J0
        self.ss_S1 = (eye(3)[:, :, newaxis] * self.S1b.trace()) - self.S1b
        self.ss_S2 = ((eye(3)[:, :, newaxis, newaxis] * self.S2b.trace())
                      - self.S2b)
        self.e_S1 = e_ikl_S_kl(self.S1b)
        self.e_S2 = e_ikl_S_kl(self.S2b)

        # Add in section inertias
        self.ss_J0 += trapz(self.section_inertia, x, axis=0)

        # Skew forms of I0 and S
        self.sk_I0 = np.einsum('imj,m ->ij ', eps_ijk, self.I0)
        self.sk_S0 = np.einsum('imj,mp->ijp', eps_ijk, self.S)

    def X(self, q):
        return self.X0 + np.einsum('hip,p', self.shapes, q)

    def Xdot(self, qd):
        return np.einsum('hip,p', self.shapes, qd)

    def small_rots(self, q):
        return np.einsum('hip,p', self.rotations, q)

    def R(self, q):
        rotations = np.einsum('hip,p', self.rotations, q)
        R = zeros((len(self.x), 3, 3))
        for i in range(len(self.x)):
            R[i] = qrot3(rotations[i]/2)
        return R

    def inertia_tensor(self, q):
        """
        Construct the inertia tensor corresponding to the modal coordinates

        .. math::

            I_{\theta\theta} = (\delta_{ij}J0_{kk} - J0_{ij})
                + [ \delta_{ij}(2 S_{kkp}) - (S_{ijp} + S_{jip}) +
                    (T_{ijp} + T_{jip}) ] \epsilon_p
                + [ \delta_{ij}(S_{kkpr} + T_{kkpr}) -
                    (S_ijpr + T_ijpr) ] \epsilon_p \epsilon_r

        """
        #inertia = eye(3)*self.J0.trace() - self.J0
        if len(self.freqs) > 0:
            S1 = np.einsum('ijp,p', self.S1, q)
            T1 = np.einsum('ijp,p', self.T1, q)
            S2 = np.einsum('ijpr,p,r', self.S2, q, q)
            T2 = np.einsum('ijpr,p,r', self.T2, q, q)
            inertia = (self.ss_J0 + eye(3)*2*S1.trace() -
                       (S1 + S1.T) - (T1 + T1.T) +
                       eye(3)*(S2.trace() + T2.trace()) - (S2 + T2))
        else:
            inertia = self.ss_J0.copy()
        return inertia

    def strain_strain(self):
        if len(self.freqs) > 0:
            A = np.einsum('mmpr', self.S2) + np.einsum('mmpr', self.T2)
        else:
            A = np.zeros((0, 0))
        return A

    def rotation_strain(self, q):
        if len(self.freqs) > 0:
            A = (self.S1 + self.T1 + np.einsum('klpr,r', self.S2, q) +
                 np.einsum('klrp,r', self.T2, q))
            B = np.einsum('ikl,lkp', eps_ijk, A)
        else:
            B = np.zeros((3, 0))
        return B

    def inertia_vel(self, q, qd):
        if len(self.freqs) > 0:
            S1 = np.einsum('ijp,p', self.S1, qd)
            T1 = np.einsum('ijp,p', self.T1, qd)
            S2 = np.einsum('ijpr,p,r', self.S2, qd, q)
            T2 = np.einsum('ijpr,p,r', self.T2, qd, q)
            A = (eye(3)*(S1 + T1).trace() - S1 - T1 +
                 eye(3)*(S2 + T2).trace() - S2 - T2)
        else:
            A = zeros((3, 3))
        return A

    def quad_stress(self, q, qd, Wp):
        if len(self.freqs) > 0:
            A1 = self.S1 + self.T1
            A2 = (np.einsum('ijpr,r', self.S2, q) +
                  np.einsum('ijrp,r', self.T2, q))

            # depending on angular velocity
            C = (np.einsum('ijp,i,j', A1+A2, Wp, Wp) -
                 dot(Wp, Wp) * ((A1+A2).trace()))

            # depending on strain velocity
            B2 = self.S2 - self.T2.swapaxes(2, 3)
            D = 2 * np.einsum('ijk,j,l,ikpl', eps_ijk, Wp, qd, B2)

            return C + D
        else:
            return np.zeros(0)

    def distributed_loading(self, P, q):
        """
        Return the generalised forces corresponding to the distributed load P.

        Returns a tuple (Q_r, Q_w, Q_e) of forces/moments/stresses.
        """
        X = self.X(q)  # length x 3
        XcrossP = np.einsum('ijk,hj,hk->hi', eps_ijk, X, P)  # length x 3
        UTP = np.einsum('hip,hi->hp', self.shapes, P)
        Qr = -trapz(P,       self.x, axis=0)
        Qw = trapz(XcrossP, self.x, axis=0)
        Qe = -trapz(UTP,     self.x, axis=0)
        return Qr, Qw, Qe

    def geometric_stiffness(self, N):
        """
        Calculate the geometric stiffening matrix corresponding to an applied
        force N.
        """
        # loop through SEGMENTS
        #Nmode = len(self.freqs)
        #Nstns = len(self.x)
        #kG = zeros((Nmode,Nmode))
        #for i in range(Nstns-1):
        #    dx = self.x[i+1] - self.x[i]
        #    for p in range(Nmode):
        #        slope_p = self.shapes[i+1,:,p] - self.shapes[i,:,p]
        #        for q in range(Nmode):
        #            slope_q = self.shapes[i+1,:,q] - self.shapes[i,:,q]
        #            intN = trapz(N[i:i+2], self.x[i:i+2], axis=0)
        #            kG[p,q] += dot(slope_p,slope_q)/dx**2 * intN

        # XXX Assume force acts axially along the beam
        dx = np.diff(self.x)
        slopes = np.r_[
            #zeros((1,)+self.shapes.shape[1:]),
            np.diff(self.shapes, axis=0) / dx[:, newaxis, newaxis],
            zeros((1,)+self.shapes.shape[1:]),
        ]  # use slope of preceding segment
        inty = np.einsum('hp,hq->hpq', slopes[:, 1, :], slopes[:, 1, :])
        intz = np.einsum('hp,hq->hpq', slopes[:, 2, :], slopes[:, 2, :])
        kG = trapz((inty + intz) * N[:, newaxis, newaxis], x=self.x, axis=0)
        return kG
