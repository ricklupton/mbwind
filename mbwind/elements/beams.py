"""
Flexible beam elements
"""
from __future__ import division
import numpy as np
from numpy import array, zeros, eye, dot, pi
from ..core import Element
from ..utils import skewmat, rotmat_x, rotmat_y, rotmat_z, qrot3, euler_param_E

# Slices to refer to parts of matrices
VP = slice(0,3)
WP = slice(3,6)
VD = slice(6,9)
WD = slice(9,12)


class UniformBeam(Element):
    _ndistal = 1
    _nstrain = 6
    _nconstraints = 6

    def __init__(self, name, length, density, EA, EIy, EIz, GIx=0.0, Jx=0.0,
                 wind=None, distal_load=None):
        '''
        Euler beam element.

         - length  : undeformed length of beam
         - density : mass per unit length
         - EA      : Extensional stiffness
         - EIy     : bending stiffness about y axis
         - EIz     : bending stiffness about z axis
         - GIx     : torsional stiffness (G*Ix*kx)

        '''
        Element.__init__(self, name)
        self.length = length
        self.linear_density = density
        self.damping = 0.1
        self.EA = EA
        self.GIx = GIx
        self.EIy = EIy
        self.EIz = EIz
        self.Jx = Jx
        self.wind = wind
        self.distal_load = distal_load
        self._initial_calcs()

    def _calc_mass_coeffs(self):
        m = self.linear_density # mass per unit length
        l0 = self.length
        # Integrals of interpolating factors
        # with t2 = l0*(xi - p1 + p2)
        # self._mass_coeffs = array([
        #     [13*l0*m/35, 11*l0**2*m/210, 9*l0*m/70, 13*l0**2*m/420],
        #     [0, l0**3*m/105, 13*l0**2*m/420, l0**3*m/140],
        #     [0, 0, 13*l0*m/35, 11*l0**2*m/210],
        #     [0, 0, 0, l0**3*m/105]
        # ])

        # with t2 = l0*(xi - p1 - p2)
        self._mass_coeffs = array([
            [13*l0*m/35, -l0**2*m/105, 9*l0*m/70, 13*l0**2*m/420],
            [0, 2*l0**3*m/105, -31*l0**2*m/420, -l0**3*m/84],
            [0, 0, 13*l0*m/35, 11*l0**2*m/210],
            [0, 0, 0, l0**3*m/105]
        ])

    def _initial_calcs(self):
        self._calc_mass_coeffs()
        # Set constant parts of mass matrix
        c = self._mass_coeffs
        I = eye(3)
        self.mass_vv[VP,VP] = c[0,0]*I
        self.mass_vv[VP,VD] = c[0,2]*I
        self.mass_vv[VD,VP] = c[0,2]*I
        self.mass_vv[VD,VD] = c[2,2]*I

        # Shape functions at distal node
        self.Sr_d = self._beam_rfuncs(1.0)
        self.Sq_d = self._beam_qfuncs(1.0)

        # Stiffness matrix
        EA = self.EA
        GIx,EIy,EIz = self.GIx, self.EIy, self.EIz
        l = self.length
        self.stiffness = array([
            [EA, 0,           0,           0,     0,          0         ],
            [0,  12*EIz/l**3, 0,           0,     0,         -6*EIz/l**2],
            [0,  0,           12*EIy/l**3, 0,     6*EIy/l**2, 0,        ],
            [0,  0,           0,           GIx/l, 0,          0         ],
            [0,  0,           6*EIy/l**2,  0,     4*EIy/l,    0         ],
            [0, -6*EIz/l**2,  0,           0,     0,          4*EIz/l   ],
        ])

    def _beam_rfuncs(self, s):
        # centreline in local coords in terms of generalised strains
        l = self.length
        return array([
            [s, 0,               0,               0, 0,             0,           ],
            [0, 3*s**2 - 2*s**3, 0,               0, 0,            -l*(s**2-s**3)],
            [0, 0,               3*s**2 - 2*s**3, 0, l*(s**2-s**3), 0            ],
        ])

    def _beam_qfuncs(self, s):
        # rotation in local coords in terms of generalised strains
        l = self.length
        return 0.5 * array([
            [0, 0,            0,            s, 0,          0          ],
            [0, 0,           -6*(s-s**2)/l, 0, 3*s**2-2*s, 0          ],
            [0, 6*(s-s**2)/l, 0,            0, 0,          3*s**2-2*s ],
        ])

    def _beam_centreline(self, s, xstrain):
        # return centreline in proximal frame
        S = self._beam_rfuncs(s)
        X0 = array([self.length*s, 0, 0])
        return X0 + dot(S, xstrain)

    def _beam_centreline_vel(self, s, vstrain):
        # return centreline in proximal frame
        S = self._beam_rfuncs(s)
        return dot(S, vstrain)

    def _beam_rotation(self, s, xstrain):
        # return cross-section rotation (Euler parameters in proximal frame)
        B = self._beam_qfuncs(s)
        q1,q2,q3 = dot(B, xstrain).flatten()
        if np.amax(np.abs([q1,q2,q3])) > 1:
            raise Exception('Euler parameters excced 1')
        q0 = np.sqrt(1.0 - q1**2 - q2**2 - q3**2)
        assert not np.isnan(q0)
        return array([q0,q1,q2,q3])

    def _beam_rotation_vel(self, s, xstrain, vstrain):
        # return cross-section rotation (Euler parameters in proximal frame)
        B = self._beam_qfuncs(s)
        qd1,qd2,qd3 = dot(B, vstrain).flatten()
        q0,q1,q2,q3 = self._beam_rotation(1.0, xstrain)
        qd0 = -(q1*qd1 + q2*qd2 + q3*qd3)/q0
        return array([qd0,qd1,qd2,qd3])

    def _beam_R(self, s, xstrain):
        # rotation matrix of cross-section in proximal frame
        B = self._beam_qfuncs(s)
        return qrot3(dot(B, xstrain))

    def calc_distal_pos(self):
        X  = array([self.length,0,0]) + self.xstrain[0:3]
        R  = qrot3(0.5*self.xstrain[3:6])
        self.rd[:] = self.rp + dot(self.Rp,X)
        self.Rd[:,:] = dot(self.Rp, R)

    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvv * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """
        X  = array([self.length,0,0]) + self.xstrain[0:3] # Distal pos in local coords
        Xd = self.vstrain[0:3]
        #wps = skewmat(self.vp[3:])

        q  = self._beam_rotation(1.0, self.xstrain)
        qd = self._beam_rotation_vel(1.0, self.xstrain, self.vstrain)
        E  = euler_param_E(q)
        Ed = euler_param_E(qd)
        wrel = 2*dot(E,qd)
        assert np.allclose(q [1:]*2, self.xstrain[3:])
        assert np.allclose(qd[1:]*2, self.vstrain[3:])

        # distal vel
        self.F_vp[VP,WP] = -dot(self.Rp,dot(skewmat(X),self.Rp.T))

        # Distal vel due to strain
        self.F_ve[VP,:] = dot(self.Rp, self.Sr_d)
        # Distal ang vel due to strain
        # TODO: is this right?
        self.F_ve[WP,:] = dot(self.Rp, dot(2*E[:,1:], self.Sq_d))

        assert not np.any(np.isnan(self.F_vp))
        assert not np.any(np.isnan(self.F_ve))

        # Quadratic velocity terms
        self.F_v2[VP] = 2*dot(self.wps, dot(self.Rp,Xd)) \
                            + dot(dot(self.wps,self.wps), dot(self.Rp, X))
        self.F_v2[WP] = 2*dot(self.Rp, dot(Ed,qd)) + dot(self.wps, dot(self.Rp, wrel))

    def shape(self, N=20):
        # proximal values
        shape = [dot(self.Rp, self._beam_centreline(i/N, self.xstrain)) for i in range(N+1)]
        return [
            self.rp + array(shape),
            self.rd + np.r_[ [dot(self.Rd,[1,0,0])], [[0,0,0]],
                             [dot(self.Rd,[0,1,0])], [[0,0,0]],
                             [dot(self.Rd,[0,0,1])] ],
            np.r_[ [self.rp], [self.rp + dot(self.Rp,[self.length,0,0])], [self.rd] ]
        ]

    shape_plot_options = [
        {'c': 'g', 'marker': 'o', 'lw': 2, 'ms': 1},
        {'c': 'k', 'lw': 0.5},
        {'c': 'k', 'lw': 2, 'alpha': 0.3},
    ]

    def calc_mass(self):
        ep = self.Rp[:,0] # unit vectors along elastic line
        ed = self.Rd[:,0]
        eps = skewmat(self.Rp[:,0]) # unit vectors along elastic line
        eds = skewmat(self.Rd[:,0])
        wps = self.wps #skewmat(self.vp[WP]) # angular velocity matrices
        wds = skewmat(self.vd[WP])

        # lumped inertia of cross-section
        # XXX check this is right - also add perpendicular inertia of cross-
        #     section, assumed to be half (ok for uniform laminar)
        Jbar = self.Jx * np.diag([1.0, 0.5, 0.5])
        Jxxp = dot(self.Rp, dot(Jbar, self.Rp.T)) / 2
        Jxxd = dot(self.Rd, dot(Jbar, self.Rd.T)) / 2

        # shorthand
        c = self._mass_coeffs
        m = self.mass_vv
        gv = self.quad_forces

        ## MASS MATRIX ##
        #[VP,VP] constant
        m[VP,WP] = -c[0,1]*eps
        #[VP,VD] constant
        m[VP,WD] = -c[0,3]*eds

        m[WP,VP] =  m[VP,WP].T
        m[WP,WP] = -c[1,1]*dot(eps,eps) + Jxxp
        m[WP,VD] =  c[1,2]*eps
        m[WP,WD] = -c[1,3]*dot(eps,eds)

        #[VD,VP] constant
        m[VD,WP] =  m[WP,VD].T
        #[VD,VD] constant
        m[VD,WD] = -c[2,3]*eds

        m[WD,VP] =  m[VP,WD].T
        m[WD,WP] =  m[WP,WD].T
        m[WD,VD] =  m[VD,WD].T
        m[WD,WD] = -c[3,3]*dot(eds,eds) + Jxxd

        ## QUADRATIC FORCES ## (remaining terms)
        gv[VP] = c[0,1]*dot(        dot(wps,wps) ,ep) + c[0,3]*dot(        dot(wds,wds) ,ed)
        gv[WP] = c[1,1]*dot(dot(eps,dot(wps,wps)),ep) + c[1,3]*dot(dot(eps,dot(wds,wds)),ed)
        gv[VD] = c[1,2]*dot(        dot(wps,wps) ,ep) + c[2,3]*dot(        dot(wds,wds) ,ed)
        gv[WD] = c[1,3]*dot(dot(eds,dot(wps,wps)),ep) + c[3,3]*dot(dot(eds,dot(wds,wds)),ed)
        # quadratic stresses are all zero

    def calc_external_loading(self):
        # Gravity loads
        self._set_gravity_force()

        # Constitutive loading
        self.applied_stress[:] = dot(self.stiffness, self.xstrain) + \
                                 dot(self.damping, dot(self.stiffness, self.vstrain))

        # Wind loading
        if self.wind:
            Cd = 2
            diameter = 2.0
            # loop through stations
            Nst = 5
            seclength = self.length / Nst
            for xi in np.linspace(0,1,num=Nst):
                # XXX This would depend on position for non-uniform wind
                windvel = self.wind.get_wind_speed(self.system.time)
                local_R = self._beam_R(xi, self.xstrain)
                Xskew = skewmat(self._beam_centreline(xi, self.xstrain))
                SX = self._beam_rfuncs(xi)
                local_windvel = dot(dot(self.Rp,local_R).T, windvel)
                local_dragforce = 0.5*1.225*Cd*diameter*(local_windvel)**2 * seclength
                # generalised forces
                Qrp = dot(dot(self.Rp,local_R), local_dragforce)
                Qwp = dot(self.Rp, dot(Xskew, dot(local_R, local_dragforce)))
                Qstrain = dot(SX.T, dot(local_R, local_dragforce))
                self.applied_forces[VP] += Qrp
                self.applied_forces[WP] += Qwp
                self.applied_stress[:]  += Qstrain

        if self.distal_load:
            global_force = self.distal_load(self.system.time)
            # generalised forces
            #Qrp = global_force[:3]
            #Qwp = dot(skewmat(self.rd - self.rp), global_force[:3]) + global_force[3:]
            #Qstrain = np.r_[ dot(self.Rp.T, global_force[:3]),
            #                 dot(self.Rp.T, global_force[3:]) ]
            Qrp = global_force
            Qwp = dot(skewmat(self.rd - self.rp), global_force[:3])
            Qstrain = dot(np.eye(6,3), dot(self.Rp.T, global_force[:3]))
            self.applied_forces[VP] += Qrp
            self.applied_forces[WP] += Qwp
            self.applied_stress[:]  += Qstrain


class TaperedBeam(UniformBeam):
    _ndistal = 1
    _nstrain = 6
    _nconstraints = 6

    def __init__(self, name, length, density, EA, EIy, EIz, GIx=0.0, Jx=0.0):
        '''
        Euler beam element.

         - length  : undeformed length of beam
         - density : mass per unit length
         - EA      : Extensional stiffness
         - EIy     : bending stiffness about y axis
         - EIz     : bending stiffness about z axis
         - GIx     : torsional stiffness (G*Ix*kx)

        '''
        Element.__init__(self, name)
        self.length = length
        self.linear_density = density
        self.damping = 0.1
        self.EA = EA
        self.GIx = GIx
        self.EIy = EIy
        self.EIz = EIz
        self.Jx = Jx
        self._initial_calcs()

    def _calc_mass_coeffs(self):
        m1, m2 = map(float, self.linear_density) # mass per unit length
        l = self.length
        self._mass_coeffs = l * array([
            [2*m1/7 + 3*m2/35, l*(m1 - 5*m2)/420, 9*m1/140 + 9*m2/140, l*(7*m1 + 6*m2)/420],
            [l*(m1 - 5*m2)/420, l**2*(5*m1 + 11*m2)/840, -l*(8*m1 + 23*m2)/420, -l**2*(m1/280 + m2/120)],
            [9*m1/140 + 9*m2/140, -l*(8*m1 + 23*m2)/420, 3*m1/35 + 2*m2/7, l*(7*m1 + 15*m2)/420],
            [l*(7*m1 + 6*m2)/420, -l**2*(m1/280 + m2/120), l*(7*m1 + 15*m2)/420, l**2*(m1/280 + m2/168)]
        ])

    def _initial_calcs(self):
        self._calc_mass_coeffs()
        # Set constant parts of mass matrix
        c = self._mass_coeffs
        I = eye(3)
        self.mass_vv[VP,VP] = c[0,0]*I
        self.mass_vv[VP,VD] = c[0,2]*I
        self.mass_vv[VD,VP] = c[0,2]*I
        self.mass_vv[VD,VD] = c[2,2]*I

        # Shape functions at distal node
        self.Sr_d = self._beam_rfuncs(1.0)
        self.Sq_d = self._beam_qfuncs(1.0)

        # Stiffness matrix
        EA = self.EA
        GIx,EIy,EIz = self.GIx, self.EIy[0], self.EIz[0]
        l = self.length
        self.stiffness = array([
            [EA, 0,           0,           0,     0,          0         ],
            [0,  12*EIz/l**3, 0,           0,     0,         -6*EIz/l**2],
            [0,  0,           12*EIy/l**3, 0,     6*EIy/l**2, 0,        ],
            [0,  0,           0,           GIx/l, 0,          0         ],
            [0,  0,           6*EIy/l**2,  0,     4*EIy/l,    0         ],
            [0, -6*EIz/l**2,  0,           0,     0,          4*EIz/l   ],
        ])

    def calc_external_loading(self):
        # Gravity loads
        self._set_gravity_force()



#==============================================================================
# class EulerBeam(UniformBeam):
#     def __init__(self, name, x, density, EA, EIy, EIz, GIx=0.0, Jx=0.0, wind=None):
#         '''
#         Euler beam element.
#
#          - x       : undeformed coordinates where beam properties are given
#          - density : mass per unit length, specified at points given in x
#          - EA      : Extensional stiffness
#          - EIy     : bending stiffness about y axis
#          - EIz     : bending stiffness about z axis
#          - GIx     : torsional stiffness (G*Ix*kx)
#
#         '''
#         Element.__init__(self, name)
#         assert x[0] == 0.0
#         self.length = x[-1]
#         self.props_x = x
#         self.linear_density = density
#         self.EA = EA
#         self.GIx = GIx
#         self.EIy = EIy
#         self.EIz = EIz
#         self.Jx = Jx
#         self.wind = wind
#         self.damping = 0.1
#         self._initial_calcs()
#
#     def _calc_mass_coeffs(self):
#         rho = self.linear_density # mass per unit length
#         x = self.props_x
#         l = self.props_x[-1]
#
#         # Integrals of interpolating factors
#         #  p1 = 3 xi^2 - 2 xi^3 = (3x^2 * l - 2x^3)/l^3
#         #  p2 =   xi^2 -   xi^3 = ( x^2 * l -  x^3)/l^3
#         p1 = (3 * l * x**2 - 2 * x**3) / l**3
#         p2 = (    l * x**2 -     x**3) / l**3
#
#         # Combined terms
#         terms = array([
#             1 - p1,
#             x - l*(p1 - p2),
#             p1,
#             l*p2,
#         ])
#
#         self._mass_coeffs = np.zeros((4,4))
#         for i in range(4):
#             for j in range(4):
#                self._mass_coeffs[i,j] = np.trapz(rho * terms[i] * terms[j], x)
#==============================================================================
