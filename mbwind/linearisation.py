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


def _create_strain_array(system, initial_values_dict):
    """
    Return an array of strains with zeros apart from the given values

    Parameters
    ----------

     - system: the System
     - initial_values_dict: dictionary of {element-name: dof-value-list}
    """

    # Dictionary of {element-name: dof-value-list} items
    z = np.zeros(len(system.q.dofs))
    for element_name, element_dofs in initial_values_dict.items():
        istrain = system.elements[element_name]._istrain
        idofs = system.qd.dofs.subset
        for i, dof in zip(istrain, element_dofs):
            try:
                z[idofs.index(i)] = dof
            except ValueError:
                print "(skipping prescribed state {})".format(i)
                pass  # this dof of the element must be prescribed
    return z


class LinearisedSystem(object):
    @classmethod
    def from_system(cls, system, z0=None, zd0=None, zdd0=None,
                    perturbation=None):
        """
        Linearise ``system`` about the point ``z0``, ``zd0``, ``zdd0``.

        If the linearisation point is not given, the current positions and
        velocities of the system, and zero acceleration, are assumed.
        """
        if perturbation is None:
            perturbation = 1e-3  # could have a better default
        assert perturbation > 0

        def _prepare_initial_values(zx, default):
            if zx is None:
                return default
            elif zx is 0:
                return 0 * default
            elif isinstance(zx, dict):
                return _create_strain_array(system, zx)
            else:
                return zx

        f = system.B.shape[0]  # number of DOFs
        z0 = _prepare_initial_values(z0, system.q.dofs[:])
        zd0 = _prepare_initial_values(zd0, system.qd.dofs[:])
        zdd0 = _prepare_initial_values(zdd0, zeros(len(system.q.dofs)))

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

        h = perturbation
        M = array([derivative(perturb, 0, h, args=(i, 2)) for i in range(f)]).T
        C = array([derivative(perturb, 0, h, args=(i, 1)) for i in range(f)]).T
        K = array([derivative(perturb, 0, h, args=(i, 0)) for i in range(f)]).T
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

        return LinearisedSystem(newM, newC, newK, self.z0, self.zd0, self.zdd0)

    def multiblade_transform2(self, azimuth_omega, iblades):
        """
        Perform a multi-blade coordinate (MBC) transform.
        -- following the NREL MBC reference
        """
        azimuth = self.z0[azimuth_omega]
        omega = self.zd0[azimuth_omega]
        Nb = len(iblades) # number of blades
        if Nb != 3:
            raise NotImplementedError("Only 3 blades implemented")
        Ndof = len(iblades[0]) # number of DOFs per blade
        if any((len(ib) != Ndof for ib in iblades)):
            raise ValueError("All blades must have same number of DOFs")

        # Transformation block matrices
        tx1 = np.ones((3, 3))
        tx2 = np.zeros((3, 3))
        tx3 = np.zeros((3, 3))
        for i in range(3):
            tx1[i, 1] = cos(azimuth + 2*i*pi/Nb)
            tx1[i, 2] = sin(azimuth + 2*i*pi/Nb)
            tx2[i, 1] = -sin(azimuth + 2*i*pi/Nb)
            tx2[i, 2] = cos(azimuth + 2*i*pi/Nb)
            tx3[i, 1] = -cos(azimuth + 2*i*pi/Nb)
            tx3[i, 2] = -sin(azimuth + 2*i*pi/Nb)

        # Assemble transformation blocks
        T1 = np.eye(self.M.shape[0])
        T2 = np.zeros_like(self.M)
        T3 = np.zeros_like(self.M)
        for i_dof in zip(*iblades):
            # loop through sets of 3 indices
            T1[np.ix_(i_dof, i_dof)] = tx1
            T2[np.ix_(i_dof, i_dof)] = tx2
            T3[np.ix_(i_dof, i_dof)] = tx3

        # Transform equations
        omega_dot = 0
        T1i = np.linalg.inv(T1)
        Mb = dot(T1i, dot(self.M, T1))
        Cb = dot(T1i, 2*omega*dot(self.M, T2) + dot(self.C, T1))
        Kb = dot(T1i, (omega**2 * dot(self.M, T3) + omega_dot*dot(self.M, T2) +
                       omega * dot(self.C, T2) + dot(self.K, T1)))
        #print self.K[10,10], Kb[10, 10]
        #import matplotlib.pyplot as plt
        #plt.matshow(dot(self.M, T3))

        return LinearisedSystem(Mb, Cb, Kb, self.z0, self.zd0, self.zdd0)

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
