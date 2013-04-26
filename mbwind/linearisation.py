# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 12:09:48 2012

Linearisation of systems

@author: Rick Lupton
"""

from __future__ import division
import sys
import time

import numpy as np
import numpy.linalg as LA
from numpy import array, zeros, eye, dot, pi, cos, sin, trapz
from numpy import newaxis as NA
import scipy.linalg
from scipy.misc import derivative
from scipy.integrate import ode, simps

from .core import System

eps_ijk = zeros((3,3,3))
eps_ijk[0,1,2] = eps_ijk[1,2,0] = eps_ijk[2,0,1] =  1
eps_ijk[2,1,0] = eps_ijk[1,0,2] = eps_ijk[0,2,1] = -1




def system_residue(system, z, zd, zdd):
    """
    Set system state to z, zd, zdd and evaluate residue.

    Residue is ..... FIXME

    Warning: changes system's state vectors.
    """
    system.q.dofs[:] = z
    system.qd.dofs[:] = zd
    system.update_kinematics()
    system.calc_projections()

    RtM = dot(system.R.T, system.lhs[np.ix_(system.iReal, system.iReal)])
    Mr = dot(RtM, system.R)
    Qr = dot(system.R.T, system.rhs[system.iReal])

    return dot(Mr, zdd) - Qr + dot(RtM, system.Sc)


class LinearisedSystem(object):
    @classmethod
    def from_system(cls, system, z0=None, zd0=None, zdd0=None):
        """
        Linearise ``system`` about the point ``z0``, ``zd0``, ``zdd0``.

        If the linearisation point is not given, the current positions and
        velocities of the system, and zero acceleration, are assumed.
        """

        f = system.B.shape[0]  # number of DOFs
        if z0 is None:
            z0 = system.q.dofs[:]
        if zd0 is None:
            zd0 = system.qd.dofs[:]
        if zdd0 is None:
            zdd0 = zeros(f)

        #def Q_func(zdd, i):
        #    self.q [self.iFreeDOF] = z0
        #    self.qd[self.iFreeDOF] = zd0
        #    return self.eval_reduced_system(hi(zdd0,zdd,i))

        def one(n, i):
            y = np.zeros(n)
            y[i] = 1
            return y

        def perturb(x, i, n):
            """
            Calculate the system residue when the i'th DOF is perturbed by
            x, referring to derivative n (0=z, 1=zd, 2=zdd)
            """
            args = [z0, zd0, zdd0]
            args[n] = args[n] + x * one(len(args[n]), i)
            return system_residue(system, *args)

        ii = range(f)
        M = array([derivative(perturb, 0, 0.1, args=(i, 2)) for i in ii]).T
        C = array([derivative(perturb, 0, 0.1, args=(i, 1)) for i in ii]).T
        K = array([derivative(perturb, 0, 0.1, args=(i, 0)) for i in ii]).T
        system.q.dofs[:] = z0
        system.qd.dofs[:] = zd0
        system.update_kinematics()

        return cls(M, C, K, z0, zd0, zdd0)

    def __init__(self, M, C, K, z0, zd0, zdd0):
        """
        Represent a linearised system about a given point
        """
        assert M.shape == C.shape == K.shape
        assert M.ndim == 2 and M.shape[0] == M.shape[1]
        assert len(z0) == len(zd0) == len(zdd0) == M.shape[0]
        self.M = M
        self.C = C
        self.K = K
        self.z0 = z0
        self.zd0 = zd0
        self.zdd0 = zdd0

    def modes(self):
        w, v = scipy.linalg.eig(self.K, self.M)
        order = np.argsort(w)
        w = np.sqrt(np.real(w[order]))
        v = v[:, order]
        return w, v

    def state_space(self):
        f = self.M.shape[0]
        A = np.r_[
            np.c_[self.M, zeros((f, f))],
            np.c_[zeros((f, f)), eye(f)]
        ]
        B = np.r_[
            np.c_[self.C, self.K],
            np.c_[-eye(f), zeros((f, f))]
        ]
        return A, B

    def multiblade_transform(self, azimuth_omega, iblades):
        """
        Perform a multi-blade coordinate (MBC) transform.
        azimuth - either a tuple (azimuth angle, angular velocity of rotor)
                  or the index of the linsys DOF corresponding to azimuth angle
        iblades - list of indices of the DOFs corresponding to each blade
        """
        if isinstance(azimuth_omega, tuple):
            azimuth, omega = azimuth_omega
        else:
            azimuth = self.z0[azimuth_omega]
            omega = self.zd0[azimuth_omega]

        Nb = len(iblades) # number of blades
        if Nb != 3:
            raise NotImplementedError("Only 3 blades implemented")
        Ndof = len(iblades[0]) # number of DOFs per blade
        if any((len(ib) != Ndof for ib in iblades)):
            raise ValueError("All blades must have same number of DOFs")
        B  = eye(self.M.shape[0])
        mu = eye(self.M.shape[0])

        # Loop through each DOF (on all blades)
        for ib,ix in enumerate(iblades):
            B[ix,iblades[0]] = 1
            B[ix,iblades[1]] = cos(azimuth + 2*ib*pi/Nb)
            B[ix,iblades[2]] = sin(azimuth + 2*ib*pi/Nb)
        mu[iblades[0],iblades[0]] = 1/3
        mu[iblades[1],iblades[1]] = 2/3
        mu[iblades[2],iblades[2]] = 2/3

        R = np.zeros_like(B)
        R[iblades[2],iblades[1]] = -omega
        R[iblades[1],iblades[2]] =  omega

        Mb = dot(mu, dot(B.T, dot(self.M, B)))
        Cb = dot(mu, dot(B.T, dot(self.C, B)))
        Kb = dot(mu, dot(B.T, dot(self.K, B)))
        newM = Mb
        newC = 2*dot(Mb,R) + Cb
        newK = dot(Mb,dot(R,R)) + dot(Cb,R) + Kb

        return LinearisedSystem((newM, newC, newK), self.z0, self.zd0, self.zdd0)

    def integrate(self, tvals, z0=None, zd0=None):
        if z0 is None: z0 = self.z0
        if zd0 is None: zd0 = self.zd0

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
