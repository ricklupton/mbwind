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
from scipy.misc import derivative
from scipy.integrate import ode

from dynamics import System

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
