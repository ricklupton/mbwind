# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 12:09:48 2012

Linearisation of systems

@author: Rick Lupton
"""

from __future__ import division
import sys, time

import numpy as np
import numpy.linalg as LA
from numpy import array, zeros, eye, dot, pi, cos, sin, trapz
from numpy import newaxis as NA
import scipy.linalg
from scipy.misc import derivative
from scipy.integrate import ode, simps

from dynamics import System, UniformBeam, skewmat

from pybladed.model import Model, ParameterNotFound

eps_ijk = zeros((3,3,3))
eps_ijk[0,1,2] = eps_ijk[1,2,0] = eps_ijk[2,0,1] =  1
eps_ijk[2,1,0] = eps_ijk[1,0,2] = eps_ijk[0,2,1] = -1

def qrot3(q):
    q1,q2,q3 = q
    q0 = np.sqrt(1.0 - q1**2 - q2**2 - q3**2)
    assert not np.isnan(q0)
    return array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)    ],
        [2*(q1*q2 + q0*q3),     1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)    ],
        [2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1),     1 - 2*(q1**2 + q2**2)],
    ])

def rotation_matrix_to_quaternions(R):
    q0 = 0.5 * np.sqrt(1 + R.trace())
    q1 =(R[2,1] - R[1,2]) / (4*q0)
    q2 =(R[0,2] - R[2,0]) / (4*q0)
    q3 =(R[1,0] - R[0,1]) / (4*q0)
    return array([q0,q1,q2,q3])

class LinearisedSystem(object):
    def __init__(self, system, z0=None, zd0=None, zdd0=None):
        """
        If system is an instance of System, linearise the system about the
        point z0, zd0, zdd0.
        If system is a tuple (M,C,K) of matrics, use these as the linearised matrices.

        If the linearisation point is not given, zero is assumed.
        """

        if isinstance(system, System):
            f = system.B.shape[0] # number of DOFs
            self.K = np.zeros((f,f))
            self.C = np.zeros((f,f))
            self.M = np.zeros((f,f))
        elif isinstance(system, tuple):
            self.M, self.C, self.K = system
            f = self.M.shape[0]
        else:
            raise TypeError('Unexpected type of system')

        if z0   is None: z0   = np.zeros(f)
        if zd0  is None: zd0  = np.zeros(f)
        if zdd0 is None: zdd0 = np.zeros(f)
        self.z0 = z0
        self.zd0 = zd0
        self.zdd0 = zdd0

        if isinstance(system, System):
            self._linearise(system)

    def _linearise(self, s):
        # Function to perturb one item in array
        def hi(x0,h,i):
            y = np.zeros_like(x0)
            y[i] = h
            return x0+y

        def z_func(x, i):
            return s.eval_residue(hi(self.z0,x,i), self.zd0, self.zdd0)
        def zd_func(x, i):
            return s.eval_residue(self.z0, hi(self.zd0,x,i), self.zdd0)
        def zdd_func(x, i):
            return s.eval_residue(self.z0, self.zd0, hi(self.zdd0,x,i))
        #def Q_func(zdd, i):
        #    self.q [self.iFreeDOF] = z0
        #    self.qd[self.iFreeDOF] = zd0
        #    return self.eval_reduced_system(hi(zdd0,zdd,i))

        for iz in range(self.M.shape[0]):
            self.K[:,iz] = derivative(z_func,   0, args=(iz,))
            self.C[:,iz] = derivative(zd_func,  0, args=(iz,))
            self.M[:,iz] = derivative(zdd_func, 0, args=(iz,))

    def modes(self):
        w,v = scipy.linalg.eig(self.K,self.M)
        order = np.argsort(w)
        w = np.sqrt(np.real(w[order]))
        v = v[:,order]
        return w,v

    def state_space(self):
        f = self.M.shape[0]
        A = np.r_[
            np.c_[self.M,       zeros((f,f))],
            np.c_[zeros((f,f)), eye(f)      ]
        ]
        B = np.r_[
            np.c_[self.C,       self.K      ],
            np.c_[-eye(f),      zeros((f,f))]
        ]
        return A,B

    def multiblade_transform(self, irot, ib1, ib2, ib3):
        """
        Perform a multi-blade coordinate (MBC) transform.
        irot    specifies the index of the DOF corresponding to rotor rotation
        ib{123} are slices specifying the indices of the DOFs corresponding to each blade
        """
        assert isinstance(ib1,slice) and isinstance(ib2,slice) and isinstance(ib3,slice)
        N = len(self.M[0,ib1])
        assert N == len(self.M[0,ib2]) == len(self.M[0,ib3])
        B  = eye(self.M.shape[0])
        mu = eye(self.M.shape[0])
        I = eye(N)
        theta = self.z0[irot]
        omega = self.zd0[irot]
        for ib,ix in enumerate([ib1,ib2,ib3]):
            B[ix,ib1] = I
            B[ix,ib2] = I * cos(theta + 2*ib*pi/3)
            B[ix,ib3] = I * sin(theta + 2*ib*pi/3)
        mu[ib1,ib1] = 1 * I / 3
        mu[ib2,ib2] = 2 * I / 3
        mu[ib3,ib3] = 2 * I / 3

        R = np.zeros_like(B)
        R[ib3,ib2] = -omega * I
        R[ib2,ib3] =  omega * I

        Mb = dot(mu, dot(B.T, dot(self.M, B)))
        Cb = dot(mu, dot(B.T, dot(self.C, B)))
        Kb = dot(mu, dot(B.T, dot(self.K, B)))
        newM = Mb
        newC = 2*dot(Mb,R) + Cb
        newK = dot(Mb,dot(R,R)) + dot(Cb,R) + Kb

        return LinearisedSystem((newM, newC, newK), self.z0, self.zd0, self.zdd0)

    def integrate(self, tvals, z0, zd0):
        f = self.M.shape[0]
        Mi = LA.inv(self.M)

        def func(ti, y):
            # y constains [z, dz/dt]
            z  = y[:f]
            zd = y[f:]
            # calculate accelerations
            zdd = -dot(Mi, dot(self.C,zd-self.zd0) + dot(self.K,z-self.z0))
            # new state vector is [strains_dot, strains_dotdot]
            return np.r_[ zd, zdd ]

        # Initial conditions
        y0 = np.r_[ z0, zd0 ]

        print 'Running simulation:',
        sys.stdout.flush()
        tstart = time.clock()

        integrator = ode(func)
        integrator.set_initial_value(y0, tvals[0])
        integrator.set_integrator("dopri5")
        result = np.nan * zeros((len(tvals), 2*f))
        result[0,:] = y0
        nprint = 20
        for it,t in enumerate(tvals[1:], start=1):
            integrator.integrate(t)
            if not integrator.successful():
                print 'stopping'
                break
            result[it,:] = integrator.y
            if (it % nprint) == 0:
                sys.stdout.write('.'); sys.stdout.flush()

        print 'done (%.1f seconds)' % (time.clock() - tstart)

        return result


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
            \boldsymbol{X}_0 + \boldsymbol{U}(\boldsymbol{X}_0) \boldsymbol{\epsilon}

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
            &= \begin{bmatrix} 0 & 0 & 0 \\ 0 & J_0 & 0 \\ 0 & 0 & J_0 \end{bmatrix}
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
                 section_inertia=None, damping=None):
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
            
        section_inertia : array (`N_x` x 2) or (`N_x`)
            Radii of gyration of cross-sections, depending on shape:
             * (`N_x` x 2): two perpendicular radii of gyration
             * (`N_x`): one radius of gyration, used for both directions

        freqs : array (`N_modes`)
            Array of modal frequencies

        damping : array (`N_modes`), default zeros
            Array of modal damping factors

        """
        assert len(x) == shapes.shape[0] == rotations.shape[0] == len(density)
        assert len(freqs) == shapes.shape[2] == rotations.shape[2]
        assert shapes.shape[1] == rotations.shape[1] == 3

        if damping is None:
            damping = zeros(len(freqs))
        
        if mass_axis is None:
            mass_axis = zeros((len(x),2))
        
        newsec = zeros((len(x),3,3))
        if section_inertia is None:
            pass
        elif section_inertia.ndim == 1:
            newsec[:,1,1] = section_inertia
            newsec[:,2,2] = section_inertia
        else:
            newsec[:,1,1] = section_inertia[0]
            newsec[:,2,2] = section_inertia[1]         
        section_inertia = newsec

        self.x = x
        self.shapes = shapes
        self.rotations = rotations
        self.freqs = freqs
        self.damping = damping
        self.density = density
        self.section_inertia = section_inertia
        self.mass_axis = mass_axis

        # Prepare integrands
        X0 = np.c_[ x, mass_axis ]
        J0 = zeros((len(x),3,3))
        S1 = zeros((len(x),3,3,len(freqs)))
        T1 = zeros((len(x),3,3,len(freqs)))
        S2 = zeros((len(x),3,3,len(freqs),len(freqs)))
        T2 = zeros((len(x),3,3,len(freqs),len(freqs)))
        H2 = zeros((len(x),3,len(freqs),len(freqs)))
        H3 = zeros((len(x),3,len(freqs),len(freqs),len(freqs)))

        for i in range(len(x)):
            J0[i] = np.outer(X0[i], X0[i])  +  section_inertia[i]
            
            S1[i] = np.einsum('j,ip', X0[i], shapes[i])
            S2[i] = np.einsum('ip,jr', shapes[i], shapes[i])
            
            T1[i,:,:,:] = np.einsum('inq,nj,qp', eps_ijk,
                                section_inertia[i], rotations[i])
            T2[i,:,:,:,:] = np.einsum('ist,jlm,ls,mp,tr', eps_ijk, eps_ijk,
                                section_inertia[i], rotations[i], rotations[i])
            
            H2[i,:,:,:  ] = np.einsum('p,q,j ->jpq',  rotations[i,1], rotations[i,1], X0[i]) + \
                            np.einsum('p,q,j ->jpq',  rotations[i,2], rotations[i,2], X0[i])
            H3[i,:,:,:,:] = np.einsum('p,q,jr->jpqr', rotations[i,1], rotations[i,1], shapes[i]) + \
                            np.einsum('p,q,jr->jpqr', rotations[i,2], rotations[i,2], shapes[i])

        # Calculate shape integrals
        self.X0 = X0
        self.mass = trapz(         density,                x, axis=0)
        self.I00  = trapz(X0     * density[:,NA],          x, axis=0)
        self.J00  = simps(J0     * density[:,NA,NA],       x, axis=0)
        self.S    = trapz(shapes * density[:,NA,NA],       x, axis=0)
        self.S1   = trapz(S1     * density[:,NA,NA,NA],    x, axis=0)
        self.T1   = trapz(T1     * density[:,NA,NA,NA],    x, axis=0)
        self.S2   = trapz(S2     * density[:,NA,NA,NA,NA], x, axis=0)
        self.T2   = trapz(T2     * density[:,NA,NA,NA,NA], x, axis=0)
        self.H2   = trapz(H2     * density[:,NA,NA,NA],    x, axis=0)
        self.H3   = trapz(H3     * density[:,NA,NA,NA,NA], x, axis=0)
        
        self.I0 = zeros(3)
        self.J0 = zeros((3,3)) # XXX neglecting non-straight mass axis
        self.I0_dist = zeros((len(x),3))
        for i in range(len(x)-2,-1,-1):
            rho1 = density[i]
            rho2 = density[i+1]
            x1,y1,z1 = X0[i]
            x2,y2,z2 = X0[i+1]
            Ix = ( (2*rho2+rho1)*x2**3 + (2*rho1+rho2)*x1**3 -
                   3*rho1*x2*x1**2 - 3*rho2*x1*x2**2 ) / (6*(x2-x1))
            Iy = (rho1*y1 + rho2*y2 - (rho1*y2+rho2*y1)/2) * (x2-x1)/3
            Iz = (rho1*z1 + rho2*z2 - (rho1*z2+rho2*z1)/2) * (x2-x1)/3
            self.I0_dist[i,:] = self.I0_dist[i+1,:] + [Ix,Iy,Iz]
            self.J0[0,0] += ( (rho1+3*rho2)*x2**4 + (rho2+3*rho1)*x1**4
                            - 4*x1*x2*(rho1*x1**2 + rho2*x2**2) ) /(12*(x2-x1))
        self.I0 = self.I0_dist[0]
        
        # describe mode shapes
        # XXX: only knows about bending modes
        self.mode_descriptions = []
        order = {0: 0, 1: 0, 2: 0}
        for p in range(len(freqs)):
            direction = np.argmax(np.abs(self.shapes[-1,:,p]))
            order[direction] += 1
            self.mode_descriptions.append('Bending towards {} mode {}'.format(
                'XYZ'[direction], order[direction]))
 
    def X(self, q):
        return self.X0 + np.einsum('hip,p', self.shapes, q)
    
    def Xdot(self, qd):
        return np.einsum('hip,p', self.shapes, qd)
 
    def R(self, q):
        rotations = np.einsum('hip,p', self.rotations, q)
        R = zeros((len(self.x),3,3))
        for i in range(len(self.x)):
            R[i] = qrot3(rotations[i]/2)
        return R
        
    def inertia_tensor(self, q):
        """
        Construct the inertia tensor corresponding to the modal coordinates
        
        .. math::
            
            I_{\theta\theta} = (\delta_{ij}J0_{kk} - J0_{ij})
                + [ \delta_{ij}(2 S_{kkp}) - (S_{ijp} + S_{jip}) + (T_{ijp} + T_{jip})] \epsilon_p
                + [ \delta_{ij}(S_{kkpr} + T_{kkpr}) - (S_ijpr + T_ijpr) ] \epsilon_p \epsilon_r

        """
        inertia = eye(3)*self.J0.trace() - self.J0        
        if len(self.freqs) > 0:
            S1 = np.einsum('ijp,p', self.S1, q)
            T1 = np.einsum('ijp,p', self.T1, q)
            S2 = np.einsum('ijpr,p,r', self.S2, q, q)
            T2 = np.einsum('ijpr,p,r', self.T2, q, q)
            inertia += eye(3)*2*S1.trace() - (S1 + S1.T) + (T1 + T1.T)
            inertia += eye(3)*(S2.trace() + T2.trace()) - (S2 + T2)
        return inertia

    def strain_strain(self):
        if len(self.freqs) > 0:
            A = np.einsum('mmpr',self.S2) + np.einsum('mmpr',self.T2)
        else:
            A = np.zeros((0,0))
        return A
    
    def rotation_strain(self, q):
        if len(self.freqs) > 0:
            A = self.S1 - self.T1 + np.einsum('klpr,p', self.S2, q) + \
                    np.einsum('klrp,p', self.T2, q)
            B = np.einsum('ikl,lkp', eps_ijk, A)
        else:
            B = np.zeros((3,0))
        return B
    
    def inertia_vel(self, q, qd):
        if len(self.freqs) > 0:
            S1 = np.einsum('ijp,p', self.S1, qd)
            T1 = np.einsum('ijp,p', self.T1, qd)
            S2 = np.einsum('ijpr,p,r', self.S2, q, qd)
            T2 = np.einsum('ijpr,p,r', self.T2, q, qd)
            A = (eye(3)*S1.trace() - S1 + T1 +
                 eye(3)*(S2 + T2).trace() - S2 - T2)
        else:
            A = zeros((3,3))
        return A
    
    def quad_stress(self, q, qd, Wp):
        if len(self.freqs) > 0:
            A1 = self.S1 - self.T1
            A2 = np.einsum('ijpr,r', self.S2, q) + \
                 np.einsum('ijrp,r', self.T2, q)
            H = np.einsum('pr,r', self.H2[0], q)
            
            # depending on angular velocity
            C = np.einsum('ijp,i,j', A1+A2, Wp, Wp) - \
                dot(Wp,Wp) * ((A1+A2).trace() + 0*H)
            
            # depending on strain velocity
            B2 = self.S2 - self.T2.swapaxes(2,3)
            D = 2 * np.einsum('ijk,j,l,ikpl', eps_ijk, Wp, qd, B2)
            
            return C + D
        else:
            return np.zeros(0)
    
    def distributed_loading(self, P, q):
        """
        Return the generalised forces corresponding to the distributed load P.
        
        Returns a tuple (Q_r, Q_w, Q_e) of forces/moments/stresses.
        """
        X = self.X(q) # length x 3
        XcrossP = np.einsum('ijk,hj,hk->hi', eps_ijk, X, P)  # length x 3
        UTP = np.einsum('hip,hi->hp', self.shapes, P)
        Qr = -trapz(P,       self.x, axis=0)
        Qw =  trapz(XcrossP, self.x, axis=0)
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
            np.diff(self.shapes, axis=0) / dx[:,None,None],
            zeros((1,)+self.shapes.shape[1:]),
        ] # use slope of preceding segment
        inty = np.einsum('hp,hq->hpq', slopes[:,1,:], slopes[:,1,:])
        intz = np.einsum('hp,hq->hpq', slopes[:,2,:], slopes[:,2,:])
        kG = trapz((inty + intz) * N[:,None,None], x=self.x, axis=0)
        return kG
                    

    def save(self, filename):
        np.savez(filename, x=self.x, shapes=self.shapes, freqs=self.freqs,
                 rotations=self.rotations, density=self.density)

    @staticmethod
    def from_file(filename):
        if not filename.lower().endswith('.npz'):
            filename += '.npz'
        npz = np.load(filename)
        return ModalRepresentation(npz['x'], npz['shapes'], npz['rotations'],
                                   npz['freqs'], npz['density'])

    @staticmethod
    def from_Bladed(filename):
        """Load modal data from Bladed .prj or .$pj file."""
        bmf = BladedModesReader(filename)
        shapes_rotations = bmf.data.transpose(2,0,1)
        permute_axes = (2,0,1,5,3,4) #
        shapes_rotations = shapes_rotations[:,permute_axes,:]
        kgyr = np.zeros_like(bmf.radii)
        #mass_axis = bmf.mass_axis
        rep = ModalRepresentation(bmf.radii, shapes_rotations[:,:3,:],
                                  shapes_rotations[:,3:,:], bmf.density,
                                  bmf.freqs, section_inertia=kgyr,
                                  damping=bmf.damping) #, mass_axis=mass_axis)
        return rep

import re

class BladedModule(object):
    def __init__(self, prj, module):
        self.module = module
        pattern = ""
        pattern = r"""
            ^[ \t]*MSTART[ \t]+{modname}\b.*\n  # MSTART as 1st word then modname
            (?:(?![ \t]*MEND).*\n)*?            # 1+ lines not starting MEND (few as poss)
        """.format(modname=self.module)
        pattern += r'''
        ^[ \t]*{kw}[ \t](.*)     # "kw" as 1st word, catch value in group
        '''.format(kw=self.keyword)
        self.regex = re.compile(pattern, re.M+re.I+re.X)

class BladedModesReader(object):
    def __init__(self, filename):
        self.data = None
        self.load(filename)

    def load(self, filename):
        self.filename = filename

        # read header file and find RMODE module
        with open(filename,'rU') as f:
            s = f.read()
        model = Model(s)

        try:
            model.get_param('MULTIBODY')
        except ParameterNotFound:
            raise NotImplemented('Only Bladed versions >= v4 are supported')

        # Blade station radii
        self.radii = model.get_param(('BGEOMMB','RJ')).split(',')
        self.radii = array(map(float, self.radii[::2])) # values duplicated for split station

        # Mass axis coordinates
        #mass_axis = model.get_param(('BGEOMMB','')).split(',')

        try:
            module = model.get_module('RMODE')
            module = re.search(r'''
                    NBLADE[ \t]+(\d+)\n
                    (?:TYPE[ \t]+(.*)\n
                    FREQ[ \t]+(.*)\n
                    DAMP[ \t]+(.*)\n
                    ((?:MD.*\n)+|CRYPT))?
                ''', module, re.VERBOSE).groups()
        except (ParameterNotFound, AttributeError):
            raise Exception("Couldn't read RMODE module from '%s'" % filename)

        nmodes = int(module[0])
        self.data = np.zeros((6, nmodes, len(self.radii)))
        if nmodes > 0:
            self.types = module[1].split()
            self.freqs = array(map(float, module[2].split()))
            self.damping = array(map(float, module[3].split()))
            rest = module[4]

            if rest == 'CRYPT':
                raise NotImplemented("Cannot read encrypted modes")

            for i,line in enumerate(rest.splitlines()):
                imode = int(i/6)
                icomp = i % 6
                assert line.startswith('MD%02d%d' % (1+imode, 1+icomp))
                d = [float(xx.strip(',')) for xx in line[5:].split()]
                self.data[icomp,imode,:] = d
        else:
            raise Exception('No modes found in "%s"' % filename)

        # extract channel names
        self.axnames = ['Component', 'Mode', 'Radius']
        self.axticks = [
            ['x', 'y', 'z', 'Rx', 'Ry', 'Rz'],
            ['%d(%s)' % xx for xx in zip(range(1,1+nmodes), self.types)],
            ['%g' % xx for xx in self.radii],
        ]
        self.axvals = [
            range(6),
            range(nmodes),
            self.radii,
        ]

        # Other data
        self.density = model.get_param(('BMASSMB','MASS')).split(',')
        self.density = array(map(float, self.density[::2])) # ignore split stations

    def _splitquoted(self, string):
        """Split on spaces, keeping quoted things together"""
        parts = re.findall(r"('[^']*'|[^\n ]+)", string)
        return [xx.replace("'", "") for xx in parts]

    def _getproperty(self, prop, string, split=False):
        """Get value of a Bladed prj file property
        Return last occurrence if more than one. Allow continuation onto new
        lines if they start with whitespace.
        """
        matches = re.findall(r'^%s[ \t]+((?:.|\n(?=[ \t]))*)' % prop,
                             string, re.MULTILINE)
        if len(matches) > 0:
            return (split and splitquoted(matches[-1]) or matches[-1])
        return None

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
