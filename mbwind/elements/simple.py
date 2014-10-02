"""
Simple elements
"""

from functools import reduce
import numpy as np
from numpy import zeros, eye, dot, pi, sin, cos
from ..base_element import Element
from ..utils import skewmat, rotmat_x, rotmat_y, rotmat_z

# Slices to refer to parts of matrices
VP = slice(0, 3)
WP = slice(3, 6)
VD = slice(6, 9)
WD = slice(9, 12)


class Hinge(Element):
    """
    A Hinge allows rotation about a specified axis. Nodes are coincident.
    """
    _nstrain = 1
    _ndistal = 1
    _nconstraints = 6

    def __init__(self, name, hinge_axis, post_transform=None):
        Element.__init__(self, name)
        self.hinge_axis = hinge_axis  # proximal node coords
        self.stiffness = 0.0
        self.damping = 0.0
        if post_transform is None:
            post_transform = np.eye(3)
        self.post_transform = post_transform
        self.internal_torque = None

    def _set_wrapping(self):
        self.system.q.wrap_levels[self.system.q.indices(self.istrain)[0]] = 2*pi

    def calc_distal_pos(self):
        vs = skewmat(self.hinge_axis)
        th = self.xstrain[0]
        Rn = np.eye(3) + vs*np.sin(th) + 2*dot(vs, vs)*np.sin(th/2)**2
        Rc = self.post_transform
        self.rd[:] = self.rp
        self.Rd[:, :] = dot(self.Rp, dot(Rn, Rc))

    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvv * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """
        n = dot(self.Rp, self.hinge_axis) # global hinge axis vector
        #wps = skewmat(self.vp[3:])            # angular velocity matrix
        thd = self.vstrain[0]             # theta dot

        self.F_ve[3:,:] = n[:,np.newaxis]
        self.F_v2[3:] = thd*dot(self.wps, n)

    def calc_external_loading(self):
        self.applied_stress[0] = (self.stiffness * self.xstrain[0] +
                                  self.damping * self.vstrain[0])
        if self.internal_torque is not None:
            if callable(self.internal_torque):
                time = self.system.time if self.system else 0
                loading = self.internal_torque(self, time)
            else:
                loading = np.asarray(self.internal_torque)
            self.applied_stress[0] += -loading  # NB minus sign


class PrismaticJoint(Element):
    _ndistal = 1
    _nstrain = 1
    _nconstraints = 6

    def __init__(self, name, axis, post_transform=None):
        Element.__init__(self, name)
        self.axis = axis  # proximal node coords
        self.stiffness = 0.0
        self.damping = 0.0
        if post_transform is None:
            post_transform = np.eye(3)
        self.post_transform = post_transform
        self.internal_force = None

    def calc_distal_pos(self):
        n = dot(self.Rp, self.axis)  # axis in global frame
        self.rd[:] = self.rp + self.xstrain[0]*n
        self.Rd[:, :] = dot(self.Rp, self.post_transform)

    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvv * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """
        n = dot(self.Rp, self.axis)  # global axis vector
        self.F_vp[0:3, 3:6] = -self.xstrain[0] * skewmat(n)
        self.F_ve[0:3, :] = n[:, np.newaxis]
        self.F_v2[0:3] = (
            2 * self.vstrain[0] * dot(self.wps, n) +
            self.xstrain[0] * dot(self.wps, dot(self.wps, n)))

    def calc_external_loading(self):
        self.applied_stress[0] = (self.stiffness * self.xstrain[0] +
                                  self.damping * self.vstrain[0])
        if self.internal_force is not None:
            if callable(self.internal_force):
                time = self.system.time if self.system else 0
                loading = self.internal_force(self, time)
            else:
                loading = np.asarray(self.internal_force)
            self.applied_stress[0] += -loading  # NB minus sign


class FreeJoint(Element):
    """
    Free join with 6 degrees of freedom: 3 translational and 3 Euler angles.

    The Euler angles are
        1. roll about the twice-rotated X axis
        2. pitch about the rotated Y axis
        3. yaw about the Z axis
    """
    _ndistal = 1
    _nstrain = 6
    _nconstraints = 6

    def __init__(self, name, stiffness=None, damping=None, post_transform=None):
        Element.__init__(self, name)
        if stiffness is None: stiffness = zeros((6,6))
        if damping is None: damping = zeros((6,6))
        self.stiffness = stiffness
        self.damping = damping
        self.post_transform = post_transform
        self.internal_forces = None

        # Constant parts of transformation matrices
        self.F_ve = eye(6)

    def calc_distal_pos(self):
        Rj = reduce(dot, (rotmat_z(self.xstrain[5]),
                          rotmat_y(self.xstrain[4]),
                          rotmat_x(self.xstrain[3])))
        if self.post_transform is not None:
            Rj = dot(Rj, self.post_transform)
        self.rd[:] = self.rp + dot(self.Rp, self.xstrain[0:3])
        self.Rd[:, :] = dot(self.Rp, Rj)

    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvv * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """
        # Angle strains to angular velocity depend on current angles.
        # Derive this by writing the axis of each Euler rotation in
        # global coordinates
        a1, a2, a3 = self.xstrain[3:6]
        b1, b2, b3 = self.vstrain[3:6]
        G = np.array([
            [cos(a3)*cos(a2), -sin(a3), 0],
            [sin(a3)*cos(a2), cos(a3), 0],
            [-sin(a2), 0, 1]
        ])
        Gdot_epsdot = np.array([
            -b1*b2*cos(a3)*sin(a2) - b1*b3*sin(a3)*cos(a2) - b2*b3*cos(a3),
            -b1*b2*sin(a3)*sin(a2) + b1*b3*cos(a3)*cos(a2) - b2*b3*sin(a3),
            -b1*b2*cos(a2),
        ])

        # Proximal nodal terms
        self.F_vp[0:3, 3:6] = -skewmat(dot(self.Rp, self.xstrain[0:3]))

        # Strain terms
        self.F_ve[0:3, 0:3] = self.Rp
        self.F_ve[3:6, 3:6] = dot(self.Rp, G)

        # Quadratic accelerations:
        # 2 𝛚̃_p 𝐑_p 𝛆̇_X + 𝛚̃_p 𝛚̃_p 𝐑_p 𝛆_X
        self.F_v2[0:3] = dot(self.wps, (
            2 * dot(self.Rp, self.vstrain[0:3]) +
            dot(self.wps, dot(self.Rp, self.xstrain[0:3]))))

        # 𝛚̃_p 𝐑_p 𝐆 𝛆̇_𝜃 + 𝐑_p 𝐆̇ 𝛆̇_𝜃
        self.F_v2[3:6] = (
            dot(self.wps, dot(self.Rp, dot(G, self.vstrain[3:6]))) +
            dot(self.Rp, Gdot_epsdot))

    def calc_external_loading(self):
        # NB applied stress is negative (designed as stiffness)
        self.applied_stress[:] = (dot(self.stiffness, self.xstrain) +
                                  dot(self.damping, self.vstrain))
        if self.internal_forces is not None:
            if callable(self.internal_forces):
                time = self.system.time if self.system else 0
                loading = self.internal_forces(self, time)
            else:
                loading = np.asarray(self.internal_forces)
            self.applied_stress += -loading  # NB minus sign


class RigidConnection(Element):
    _ndistal = 1
    _nstrain = 0
    _nconstraints = 6

    def __init__(self, name, offset=None, rotation=None):
        Element.__init__(self, name)
        if offset is None:
            offset = zeros(3)
        if rotation is None:
            rotation = eye(3)
        self.offset = offset
        self.rotation = rotation

        # cache
        self.skew_offset = skewmat(self.offset)

    def calc_distal_pos(self):
        self.rd[:] = self.rp + dot(self.Rp, self.offset)
        self.Rd[:,:] = dot(self.Rp, self.rotation)

    def calc_kinematics(self):
        """
        Update kinematic transforms: F_vp, F_ve and F_v2
        [vd wd] = Fvv * [vp wp]  +  Fve * [vstrain]  +  Fv2
        """

        #wp = self.vp[3:]
        xcs = dot(self.Rp, dot(self.skew_offset, self.Rp.T))
        # Distal velocity rd_d = rd_p - xc \times w_p
        self.F_vp[0:3,3:6] = -xcs #skewmat(xc)
        # Distal velocity quadratic term w_p \times w_p \times xc
        #self.F_v2[0:3] = dot(np.outer(wp,wp), xc) - xc*dot(wp,wp)
        self.F_v2[0:3] = dot( dot(self.wps,self.wps), dot(self.Rp, self.offset) )


class RigidBody(Element):
    _ndistal = 0
    _nstrain = 0
    _nconstraints = 0

    def __init__(self, name, mass, inertia=None, Xc=None, nodal_load=None):
        '''
        Rigid body element with only one node.
        Defined by mass and inertia tensor.
        '''
        Element.__init__(self, name)
        if Xc is None: Xc = zeros(3)
        if inertia is None: inertia = zeros((3,3))
        self.mass = mass
        self.inertia = inertia
        self.Xc = Xc
        # Set constant parts of mass matrix
        self.mass_vv[VP,VP] = self.mass * eye(3)

        self.nodal_load = nodal_load

    def calc_mass(self):
        # global offset to centre of mass
        xc = dot(self.Rp, self.Xc)
        Jp = dot(self.Rp, dot(self.inertia, self.Rp.T))
        #wps = skewmat(self.vp[3:])

        ## MASS MATRIX ##
        #    mass_vv[VP,VP] constant
        self.mass_vv[VP,WP] = -self.mass * skewmat(xc)
        self.mass_vv[WP,VP] =  self.mass * skewmat(xc)
        self.mass_vv[WP,WP] =  Jp

        ## QUADRATIC FORCES ## (remaining terms)
        self.quad_forces[VP] = self.mass * dot(dot(self.wps,self.wps), xc)
        self.quad_forces[WP] = dot(self.wps, dot(Jp, self.vp[3:]))

    def calc_external_loading(self):
        self._set_gravity_force()
        if self.nodal_load is not None:
            time = self.system.time if self.system else 0
            if callable(self.nodal_load):
                global_force = self.nodal_load(self, time)
            else:
                global_force = self.nodal_load
            self.applied_forces[VP] += global_force
