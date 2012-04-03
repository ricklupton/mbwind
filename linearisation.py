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
from numpy import array, zeros, eye, dot, pi, cos, sin
import scipy.linalg
from scipy.misc import derivative
from scipy.integrate import ode

from dynamics import System, UniformBeam, skewmat

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

        print 'done'
        print '%.1f seconds elapsed' % (time.clock() - tstart)

        return result


class ModalRepresentation(object):
    def __init__(self, x, shapes, rotations, freqs, density):
        '''
        Setup modal representation for a 1-dimensional element.

        Parameters
        ----------
        x : array (`N_x`)
            Coordinates where mode shapes and densities are defined

        shapes : array (`N_x` x 3 x `N_modes`)
            Array of mode shapes defined at points in `x`

        rotations : array (`N_x` x 3 x `N_modes`)
            Array of mode rotations defined at points in `x`

        freqs : array (`N_modes`)
            Array of modal frequencies

        density : array (`N_x`)
            Linear density distribution at points defined in `x`

        '''
        assert len(x) == shapes.shape[0] == rotations.shape[0] == len(density)
        assert len(freqs) == shapes.shape[2] == rotations.shape[2]
        assert shapes.shape[1] == rotations.shape[1] == 3

        self.x = x
        self.shapes = shapes
        self.rotations = rotations
        self.freqs = freqs
        self.density = density

        # Calculate shape integrals
        self.mass      = np.trapz(    density, x)
        self.moment1   = np.trapz(x * density, x)
        self.shape_int = np.trapz(shapes, x, axis=0)

        #X0 = zeros((len(x),3))
        #X0[:,0] = x
        X0s2 = zeros((len(x),3,3))
        T2   = zeros((len(x),3,3,len(freqs)))
        S2   = zeros((len(x),3,3,len(freqs),len(freqs)))
        for i in range(len(x)):
            skew = skewmat([x[i],0,0])
            X0s2[i,:,:] = -density[i] * dot(skew, skew)

            for j in range(len(freqs)):
                T2[i,:,:,j] = -density[i] * dot(skew, shapes[i,:,j])
                for k in range(len(freqs)):
                    S2[i,:,:,j,k] = -density[i] * dot(skewmat(shapes[i,:,j]),
                                                      skewmat(shapes[i,:,k]))

        self.inertia_tensor = np.trapz(X0s2, x, axis=0)
        self.shape_int1 = np.trapz(T2, x, axis=0)
        self.shape_int2 = np.trapz(S2, x, axis=0)

    def save(self, filename):
        np.savez(filename, x=self.x, shapes=self.shapes, freqs=self.freqs,
                 rotations=self.rotations, density=self.density)

    @classmethod
    def from_file(self, filename):
        if not filename.lower().endswith('.npz'):
            filename += '.npz'
        npz = np.load(filename)
        return ModalRepresentation(npz['x'], npz['shapes'], npz['rotations'],
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
