# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:21:14 2012

@author: Rick Lupton
"""

# cython: profile=True

cimport numpy as np
import numpy as np

cdef inline rotvec(np.ndarray[double, ndim=2] R, np.ndarray[double, ndim=1] x):
    cdef np.ndarray[double, ndim=1] y = np.zeros_like(x)
    cdef Py_ssize_t i, j
    for i in range(3):
        for j in range(3):
            y[i] += R[i,j] * x[j]
    return y

cdef inline rottensor(np.ndarray[double, ndim=2] R, np.ndarray[double, ndim=2] T):
    cdef np.ndarray[double, ndim=2] Y = np.zeros_like(T)
    cdef Py_ssize_t i, j, ii, jj
    for i in range(3):
        for j in range(3):
            for ii in range(3):
                for jj in range(3):
                    Y[i,j] += T[ii,jj] * R[i,ii] * R[j,jj]
    return Y

def calc_mass(rep not None,
              np.ndarray[double, ndim=2] R not None,
              np.ndarray[double, ndim=1] w not None,
              np.ndarray[double, ndim=1] z not None,
              np.ndarray[double, ndim=1] zd not None,
              np.ndarray[double, ndim=2] Mv not None,
              np.ndarray[double, ndim=2] Me not None,
              np.ndarray[double, ndim=1] gv not None,
              np.ndarray[double, ndim=1] ge not None):
    """
    Calculate all inertial terms (mass matrix and quadratic forces)
    """
    cdef np.ndarray[double, ndim=2] sk_I0 = rep.sk_I0
    cdef np.ndarray[double, ndim=2]    S  = rep.S
    cdef np.ndarray[double, ndim=3] sk_S0 = rep.sk_S0
    cdef np.ndarray[double, ndim=2] ss_J  = rep.ss_J0
    cdef np.ndarray[double, ndim=2]  e_S1 = rep.e_S1
    cdef np.ndarray[double, ndim=3] ss_S1 = rep.ss_S1
    cdef np.ndarray[double, ndim=3]  e_S2 = rep.e_S2
    cdef np.ndarray[double, ndim=4] ss_S2 = rep.ss_S2
    cdef np.ndarray[double, ndim=4]    S2 = rep.S2
    
    cdef np.ndarray[double, ndim=2] I_ww = np.zeros((3,3)), I_rw = np.zeros((3,3))
    cdef np.ndarray[double, ndim=2] A = np.zeros((3,3))
    cdef np.ndarray[double, ndim=1] y = np.zeros(3), w_local = np.zeros(3)
    cdef int i,j,p,q,N
    cdef double x
    
    N = ss_S1.shape[2]
    
    # Prelim calculations:
    # a) inertia tensor
    for i in range(3):
        for j in range(3):
            I_ww[i,j] = ss_J[i,j]
            for p in range(N):
                I_ww[i,j] += (ss_S1[i,j,p] + ss_S1[j,i,p]) * z[p]
                for q in range(N):
                    I_ww[i,j] += ss_S2[i,j,p,q] * z[p] * z[q]
    I_ww = rottensor(R, I_ww) # rotate into global coordinates
    
    # b) translation-rotation coupling matrix
    #     I_rw = skew[ I0 + Se ]
    for i in range(3):
        for j in range(3):
            I_rw[i,j] = sk_I0[i,j]
            for p in range(N):
                I_rw[i,j] += sk_S0[i,j,p] * z[p]
    I_rw = rottensor(R, I_rw) # rotate into global coordinates
    
    # c) local angular velocity
    w_local = rotvec(R.T, w)
    
    # ---- MASS MATRIX ---- #
    Mv[:] = 0
    Me[:] = 0
    
    # Translation-translation
    Mv[0,0] = Mv[1,1] = Mv[2,2] = rep.mass
    
    # Translation-rotation
    for i in range(3):
        for j in range(3):
            Mv[i,3+j] = -I_rw[i,j]
            Mv[3+j,i] = -I_rw[i,j]

    # Translation-strain
    for i in range(3):
        for p in range(N):
            for j in range(3):
                Me[i,p] += R[i,j] * S[j,p]
    
    # Rotation-rotation
    for i in range(3):
        for j in range(3):
            Mv[3+i,3+j] = I_ww[i,j]
    
    # Rotation-strain
    for p in range(N):
        for i in range(3):
            y[i] = e_S1[i,p]
            for q in range(N):
                y[i] += e_S2[i,p,q] * z[q]
        # rotate
        y = rotvec(R, y)
        for i in range(3):
            Me[3+i,p] = -y[i]
    
    # ---- OTHER INERTIAL FORCES ---- #
    gv[:] = 0
    ge[:] = 0
    
    # Translation
    for i in range(3):
        y[i] = 0.0
        for j in range(3):
            # Centrifugal force
            y[i] += -I_rw[i,j] * w[j]
        for p in range(N):
            # Coriolis force
            y[i] += 2 * Me[i,p] * zd[p] # Me in translation - strain is R*S
    gv[0] = (w[1]*y[2] - w[2]*y[1])
    gv[1] = (w[2]*y[0] - w[0]*y[2])
    gv[2] = (w[0]*y[1] - w[1]*y[0])
    
    # Centrifugal force - rotation
    y = rotvec(I_ww, w)
    gv[3] = (w[1]*y[2] - w[2]*y[1])
    gv[4] = (w[2]*y[0] - w[0]*y[2])
    gv[5] = (w[0]*y[1] - w[1]*y[0])    

    # Centrifugal force - strain
    for p in range(N):
        for i in range(3):
            for j in range(3):
                x = ss_S1[i,j,p]
                for q in range(N):
                    x += ss_S2[i,j,p,q] * z[q]
                ge[p] += -x * w_local[i] * w_local[j]
    
    # Rotation - Coriolis force
    for i in range(3):
        y[i] = 0.0
        for j in range(3):
            for p in range(N):
                y[i] += w_local[j] * (ss_S1[i,j,p] * zd[p])
                for q in range(N):
                    y[i] += w_local[j] * (ss_S2[i,j,p,q] * zd[p] * z[q])
    y = rotvec(R, y)
    for i in range(3):
        gv[3+i] += 2 * y[i]
        
    # Strain - Coriolis force
    for p in range(N):
        for i in range(3):
            for q in range(N):
                ge[p] += -2 * e_S2[i,p,q] * zd[q] * w_local[i]
    
def inertia_tensor(rep not None, np.ndarray[double, ndim=1] z not None):
    """
    Construct the inertia tensor corresponding to the modal coordinates
    
    .. math::
        
        I_{\theta\theta} = (\delta_{ij}J0_{kk} - J0_{ij})
            + [ \delta_{ij}(2 S_{kkp}) - (S_{ijp} + S_{jip}) + (T_{ijp} + T_{jip})] \epsilon_p
            + [ \delta_{ij}(S_{kkpr} + T_{kkpr}) - (S_ijpr + T_ijpr) ] \epsilon_p \epsilon_r

    """
    cdef np.ndarray[double, ndim=2] inertia
    cdef np.ndarray[double, ndim=3] S1, T1
    cdef np.ndarray[double, ndim=4] S2, T2
    cdef int i,j,p,q,N
    cdef double x1, x2, x3, zp, zq
    
    S1 = rep.S1
    S2 = rep.S2
    T1 = rep.T1
    T2 = rep.T2
    inertia = rep.ss_J0.copy()
    N = S1.shape[-1]
    for p in range(N):
        zp = z[p]
        for i in range(3):
            # 2I * trace(S1 q)
            x1 = S1[0,0,p]
            x2 = S1[1,1,p]
            x3 = S1[2,2,p]
            inertia[i,i] += 2*(x1+x2+x3)*zp
            for j in range(3):
                # -(S1 + S1.T)q
                inertia[i,j] += -(S1[i,j,p] + S1[j,i,p])*zp
                inertia[i,j] +=  (T1[i,j,p] + T1[j,i,p])*zp
        
        for q in range(N):
            zq = z[q]
            for i in range(3):
                # I * trace(S2 q q)
                inertia[i,i] += (S2[0,0,p,q] + S2[1,1,p,q] + S2[2,2,p,q])*zp*zq
                for j in range(3):
                    # -(S2 q q) - (T2 q q)
                    inertia[i,j] += -(S2[i,j,p,q] + T2[j,i,p,q]) * zp * zq
                
    return inertia

#def strain_strain(self):
#    if len(self.freqs) > 0:
#        A = np.einsum('mmpr',self.S2) + np.einsum('mmpr',self.T2)
#    else:
#        A = np.zeros((0,0))
#    return A
#
def rotation_strain(rep not None, np.ndarray[double, ndim=1] z not None):
    
    cdef np.ndarray[double, ndim=2] result, A
    cdef np.ndarray[double, ndim=3] S1, T1
    cdef np.ndarray[double, ndim=4] S2, T2
    cdef int i,j,p,q,N
    
    # Shortcuts to shape functions
    S1 = rep.S1
    S2 = rep.S2
    T1 = rep.T1
    T2 = rep.T2
    N = S1.shape[2]
    result = np.zeros((3,N))
    A = np.zeros((3,3))
    
    for p in range(N):
        for i in range(3):
            for j in range(3):
                A[i,j] = S1[i,j,p] - T1[i,j,p]
                for q in range(N):
                    # NB T2 swapped
                    A[i,j] += (S2[i,j,q,p] + T2[i,j,p,q]) * z[q]
        
        # result_i = eps_ijk Ap_kj
        result[0,p] = -A[1,2] + A[2,1] 
        result[1,p] = -A[2,0] + A[0,2]
        result[2,p] = -A[0,1] + A[1,0]
    
    return result

def inertia_vel(rep not None, np.ndarray[double, ndim=1] z not None,
                np.ndarray[double, ndim=1] zd not None):
    
    cdef np.ndarray[double, ndim=2] result, A
    cdef np.ndarray[double, ndim=3] S1, T1
    cdef np.ndarray[double, ndim=4] S2, T2
    cdef int i,j,p,q,N
    cdef double trace = 0.0
    
    # Shortcuts to shape functions
    S1 = rep.S1
    S2 = rep.S2
    T1 = rep.T1
    T2 = rep.T2
    N = S1.shape[2]
    result = np.zeros((3,3))
    A = np.zeros((3,3))
    
    for i in range(3):
        for j in range(3):
            for p in range(N):
                # part with trace added
                A[i,j] += S1[i,j,p] * zd[p]
                # T1 part goes in as is
                result[i,j] += T1[i,j,p] * zd[p]
                for q in range(N):
                    A[i,j] += (S2[i,j,p,q] + T2[i,j,p,q]) * z[p] * zd[q]
        trace += A[i,i]

    # Add trace to diagonal and subtract A
    for i in range(3):
        result[i,i] += trace
        for j in range(3):
            result[i,j] -= A[i,j]

    return result

def quad_stress(rep not None, np.ndarray[double, ndim=1] z not None,
                np.ndarray[double, ndim=1] zd not None,
                np.ndarray[double, ndim=1] Wp not None):

    cdef np.ndarray[double, ndim=1] stress, A
    cdef np.ndarray[double, ndim=2] B
    cdef np.ndarray[double, ndim=3] S1, T1
    cdef np.ndarray[double, ndim=4] S2, T2
    cdef int i,j,p,q,N
    cdef double A_trace, Wp_Wp
    
    # Shortcuts to shape functions
    S1 = rep.S1
    S2 = rep.S2
    T1 = rep.T1
    T2 = rep.T2
    N = S1.shape[2]
    
    stress = np.zeros(N)
    A = np.zeros(3)
    B = np.zeros((3,3))
    
    Wp_Wp = np.dot(Wp,Wp)

    for p in range(N):
        # Part depending on angular velocity
        A_trace = 0.0
        for i in range(3):
            for j in range(3):
                # A1 = S1 - T1
                A[j] = S1[i,j,p] - T1[i,j,p]
                for q in range(N):
                    # A2 = (S2 q) + (T2r qr)
                    # XXX NB T2 axes swapped
                    A[j] += (S2[i,j,p,q] + T2[i,j,q,p]) * z[q]
            
                # Multiply by angular velocity:
                # C = Wp*(A1+A2)*Wp - Wp**2*trace(A1+A2)
                stress[p] += A[j] * Wp[i]*Wp[j]
                
            A_trace += A[i]
        stress[p] += -Wp_Wp * A_trace
        
        # Part depending on strain velocity 
        for i in range(3):
            for j in range(3):
                B[i,j] = 0.0
                for q in range(N):
                    # B2: NB T2 axes swapped
                    B[i,j] += 2*(S2[i,j,p,q] - T2[i,j,q,p]) * zd[q]
        # eps_ijk W_j B_ikp
        stress[p] += (B[0,2]-B[2,0])*Wp[1] + (B[1,0]-B[0,1])*Wp[2] + (B[2,1]-B[1,2])*Wp[0]
        
    return stress

#def distributed_loading(self, P, q):
#    """
#    Return the generalised forces corresponding to the distributed load P.
#    
#    Returns a tuple (Q_r, Q_w, Q_e) of forces/moments/stresses.
#    """
#    X = self.X(q) # length x 3
#    XcrossP = np.einsum('ijk,hj,hk->hi', eps_ijk, X, P)  # length x 3
#    UTP = np.einsum('hip,hi->hp', self.shapes, P)
#    Qr = -trapz(P,       self.x, axis=0)
#    Qw =  trapz(XcrossP, self.x, axis=0)
#    Qe = -trapz(UTP,     self.x, axis=0)
#    return Qr, Qw, Qe
