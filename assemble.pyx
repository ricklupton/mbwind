# cython: profile=True

cimport numpy as np

def assemble(elements, np.ndarray[double, ndim=2] mass, np.ndarray[double, ndim=1] rhs):
    """Copy matrices from elements to system"""
    cdef int i, I, j, J
    cdef np.ndarray[double, ndim=2] el_m, el_D
    cdef np.ndarray[double, ndim=1] el_g, el_f, el_s, el_F2

    for el in elements:
        el_g = el._quadratic_forces
        el_m = el._mass_matrix
        el_F2 = el.F_v2
        el_D = el._constraint_matrix
        el_f = el._applied_forces
        el_s = el._applied_stresses
        # Mass terms
        imass = el._imass()
        for I,i in enumerate(imass):
            rhs[i] += el_g[I] 
            for J,j in enumerate(imass):
                mass[i,j] += el_m[I,J]

        # Constraint terms
        for I,i in enumerate(el.imult):
            rhs[i] += el_F2[I]
            for J,j in enumerate(imass):
                mass[i,j] += el_D[I,J]
                mass[j,i] += el_D[I,J]

        # Strain terms
        for I,i in enumerate(el._inodes()):
            rhs[i] += el_f[I]
        for I,i in enumerate(el.istrain):
            rhs[i] += -el_s[I]

