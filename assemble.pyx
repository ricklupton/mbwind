# cython: profile=True

cimport numpy as np

def assemble(elements, np.ndarray[double, ndim=2] mass, np.ndarray[double, ndim=1] rhs):
    """Copy matrices from elements to system"""
    cdef int i, I, j, J
    cdef np.ndarray[double, ndim=2] el_mvv, el_mve, el_mee, el_Fvp, el_Fvd, el_Fve
    cdef np.ndarray[double, ndim=1] el_g, el_f, el_s, el_F2

    for el in elements:
        el_g = el._quadratic_forces
        el_f = el._applied_forces
        el_s = el._applied_stresses
        
        # Nodal terms
        inode = el._inodes()
        istrain = el.istrain
        el_mvv = el.mass_vv
        el_mve = el.mass_ve
        el_mee = el.mass_ee
        for I,i in enumerate(inode):
            rhs[i] += el_g[I] + el_f[I]
            
            # Node-node
            for J,j in enumerate(inode):
                mass[i,j] += el_mvv[I,J]
            
            # Node-strain
            for J,j in enumerate(istrain):
                mass[i,j] += el_mve[I,J]
                mass[j,i] += el_mve[I,J]
        
        # Strain-strain
        for I,i in enumerate(istrain):
            rhs[i] += el_g[I] - el_s[I]
            for J,j in enumerate(istrain):
                mass[i,j] += el_mee[I,J]
            
        # Constraints
        el_Fvp = el.F_vp
        el_Fvd = el.F_vd
        el_Fve = el.F_ve
        el_F2 = el.F_v2
        imass = inode + istrain
        for I,i in enumerate(el.imult):
            rhs[i] += -el_F2[I]
            for J,j in enumerate(el.iprox):
                mass[i,j] += el_Fvp[I,J]
                mass[j,i] += el_Fvp[I,J]
            for J,j in enumerate(el.idist):
                mass[i,j] += el_Fvd[I,J]
                mass[j,i] += el_Fvd[I,J]
            for J,j in enumerate(el.istrain):
                mass[i,j] += el_Fve[I,J]
                mass[j,i] += el_Fve[I,J]

