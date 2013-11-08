import numpy as np
from numpy import array, dot, pi
from mbwind.elements import (FreeJoint, RigidConnection, RigidBody, Hinge,
                             ModalElement, DistalModalElementFromScratch)


# class PrismaticJointView(object):
#     def shape(self, joint):
#         # proximal values
#         return [
#             array([joint.rd]),
#             joint.rd + np.r_[[dot(joint.Rd, [1, 0, 0])], [[0, 0, 0]],
#                              [dot(joint.Rd, [0, 1, 0])], [[0, 0, 0]],
#                              [dot(joint.Rd, [0, 0, 1])]]
#         ]

#     shape_plot_options = [
#         {'marker': 's', 'ms': 4, 'c': 'r'},
#         {'c': 'k', 'lw': 0.5}
#     ]

from matplotlib import animation
from matplotlib import pyplot as plt


def anim_mode(system, frequency, modeshape, xlim=None, ylim=None, zlim=None,
              rotor_speed=None, rotor_element=None, blade_elements=None,
              figsize=(6, 4), filename=None, save_opts=None, title=None):
    Ndof = len(modeshape)
    assert Ndof == len(system.q.dofs)

    # Choose time vector so there are at least 50 steps, or more if
    # needed so that the rotor rotation isn't too coarse.
    max_t = 2*pi/frequency
    dt = max_t/50
    if rotor_speed is not None and (rotor_speed/frequency < 5):
        max_t = 2*pi/min(frequency, rotor_speed)
        dt = min(dt, 2*pi/rotor_speed/200)
    t = np.arange(0, max_t, dt)

    # Save initial values
    initial_q = system.q.dofs[:]

    # Calculate axis limits automatically
    def nodal_positions(system, z):
        system.q.dofs[:] = z
        system.update_kinematics(calculate_matrices=False)
        qnodes = system.q.by_type(('ground', 'node'))
        # Pick out positions (not rotation matrices)
        xyz = qnodes[[i for i in range(len(qnodes)) if (i % 12) < 3]] \
            .reshape((-1, 3))
        # Add in ModalElement final station positions as a workaround
        modal_xyz = [e.station_positions()[-1] for e in system.iter_elements()
                     if isinstance(e, ModalElement)]
        return np.r_[xyz, modal_xyz]

    # Calculate nodal positions at start and middle of oscillation
    qn_a = nodal_positions(system, modeshape.real)
    qn_b = nodal_positions(system, modeshape.imag)

    # Minimum and maximum nodal positions along each axis (xyz)
    min_extent = np.minimum(qn_a, qn_b).min(axis=0) * 1.1
    max_extent = np.maximum(qn_a, qn_b).max(axis=0) * 1.1
    scale = (max_extent - min_extent).max() / 7  # a representative scale
    if xlim is None:
        xlim = (min(-scale, min_extent[0]), max(scale, max_extent[0]))
    if ylim is None:
        ylim = (min(-scale, min_extent[1]), max(scale, max_extent[1]))
    if zlim is None:
        zlim = (min(-scale, min_extent[2]), max(scale, max_extent[2]))

    # Figure out how much the nodes moved so we know how much to scale the mode
    node_motion = np.sqrt(np.sum((qn_b - qn_a)**2, axis=1)).max()
    modeshape_scale = 0.4 * scale / (node_motion if node_motion > 0 else 1.0)

    # If there is a rotating rotor, the blade coordinates will need to
    # be calculated at every timestep
    if rotor_speed is None:
        assert rotor_element is None and blade_elements is None
        def dofs_at_time(t):
            return modeshape_scale * (modeshape * np.exp(1j*frequency*t)).real
    else:
        def dofs_at_time(t):
            az = (rotor_speed * t) % (2*pi)
            z = system.convert_mbc_dofs_to_blade(modeshape, blade_elements, az)
            dofs = modeshape_scale * (z * np.exp(1j*frequency*t)).real
            dofs[system.dof_index(rotor_element, 0)] += az
            return dofs

    # Create axes: XZ, YZ, XY
    directions = ('xz', 'yz', 'xy')
    limits = {'x': xlim, 'y': ylim, 'z': zlim}
    fig, axes = plt.subplots(1, 3, figsize=figsize,
                             subplot_kw=dict(aspect='equal'))

    fig.suptitle('[%.4f rad/s] %s' % (frequency, title or ''))

    # Set up axes and views
    sysviews = []
    for ax, direction in zip(axes, directions):
        sysviews.append(SystemView(ax, direction, system))

        if limits[direction[0]] is not None:
            ax.set_xlim(limits[direction[0]])
        if limits[direction[1]] is not None:
            ax.set_ylim(limits[direction[1]])
        ax.set_xlabel(direction[0])
        ax.set_ylabel(direction[1])
        ax.grid()

    # Animation functions
    def init():
        artists = []
        for sysview in sysviews:
            sysview.reset()
            artists.extend(sysview.artists)
        return artists

    def animate(i):
        # Update system
        system.q.dofs[:] = dofs_at_time(t[i])
        system.update_kinematics(calculate_matrices=False)

        # Update artists
        artists = []
        for sysview in sysviews:
            sysview.update()
            artists.extend(sysview.artists)
        return artists

    repeat = (filename is None)
    anim = animation.FuncAnimation(fig, animate, frames=len(t), repeat=repeat,
                                   interval=10, blit=True, init_func=init)
    if filename is not None:
        if save_opts is None:
            save_opts = {}
        save_opts.update({
            'fps': 30,
            'codec': 'libtheora',
            'bitrate': 600,
            'dpi': 100,
        })
        anim.save(filename, **save_opts)
        plt.close(fig)

    # Put back system how it started
    system.q.dofs[:] = initial_q
    system.update_kinematics(calculate_matrices=False)

    return anim


from IPython.display import HTML


def display_video(filename):
    h = """
    <video width="600" height="400" controls loop>
    <source src="files/%s" type="video/ogg">
    Your browser does not support the video tag :(
    </video>
    """ % filename
    return HTML(h)


class SystemView(object):
    def __init__(self, axes, direction, system):
        self.system = system
        self.views = []
        for element in system.iter_elements():
            view_class = ElementView.get_view_for_element(element)
            self.views.append(view_class(axes, direction, element))

        self.artists = []
        for view in self.views:
            self.artists.extend(view.artists)

    def reset(self):
        for view in self.views:
            view.reset()

    def update(self):
        for view in self.views:
            view.update()


class ElementView(object):
    def __init__(self, axes, direction, element):
        self.axes = axes
        self.x = 'xyz'.index(direction[0])
        self.y = 'xyz'.index(direction[1])
        self.element = element
        self.artists = self.create_artists()

    @classmethod
    def get_view_for_element(cls, element):
        for subclass in cls.__subclasses__():
            if subclass.element is element.__class__:
                return subclass
        raise KeyError("No view defined for %s" % element.__class__)


class FreeJointView(ElementView):
    element = FreeJoint

    def create_artists(self):
        self.rd, = self.axes.plot([], [], marker='s', ms=4, c='r')
        self.Rd, = self.axes.plot([], [], lw=0.5, c='k')
        return (self.rd, self.Rd)

    def reset(self):
        self.rd.set_data([], [])
        self.Rd.set_data([], [])

    def update(self):
        e = self.element
        self.rd.set_data(e.rd[self.x], e.rd[self.y])
        orient = e.rd + np.r_[[dot(e.Rd, [1, 0, 0])], [[0, 0, 0]],
                              [dot(e.Rd, [0, 1, 0])], [[0, 0, 0]],
                              [dot(e.Rd, [0, 0, 1])]]
        self.Rd.set_data(orient[:, self.x], orient[:, self.y])


class HingeView(ElementView):
    element = Hinge

    def create_artists(self):
        self.rd, = self.axes.plot([], [], marker='s', ms=4, c='r')
        self.Rd, = self.axes.plot([], [], lw=0.5, c='k')
        return (self.rd, self.Rd)

    def reset(self):
        self.rd.set_data([], [])
        self.Rd.set_data([], [])

    def update(self):
        e = self.element
        self.rd.set_data(e.rp[self.x], e.rd[self.y])
        orient = e.rd + np.r_[[dot(e.Rd, [1, 0, 0])], [[0, 0, 0]],
                              [dot(e.Rd, [0, 1, 0])], [[0, 0, 0]],
                              [dot(e.Rd, [0, 0, 1])]]
        self.Rd.set_data(orient[:, self.x], orient[:, self.y])


class RigidConnectionView(ElementView):
    element = RigidConnection

    def create_artists(self):
        self.line, = self.axes.plot([], [], c='m', lw=1, marker='^', ms=3)
        self.distal_orientation, = self.axes.plot([], [], c='k', lw=0.5)
        return (self.line, self.distal_orientation)

    def reset(self):
        self.line.set_data([], [])
        self.distal_orientation.set_data([], [])

    def update(self):
        e = self.element
        line = np.r_[[e.rp], [e.rd]]
        self.line.set_data(line[:, self.x], line[:, self.y])
        orient = e.rd + np.r_[[dot(e.Rd, [1, 0, 0])], [[0, 0, 0]],
                              [dot(e.Rd, [0, 1, 0])], [[0, 0, 0]],
                              [dot(e.Rd, [0, 0, 1])]]
        self.distal_orientation.set_data(orient[:, self.x], orient[:, self.y])


class RigidBodyView(ElementView):
    element = RigidBody

    def create_artists(self):
        self.line, = self.axes.plot([], [], c='y', marker='x', ms=4, mew=2)
        self.rg, = self.axes.plot([], [], c='y', marker='x', ms=10, mew=3)
        return (self.line, self.rg)

    def reset(self):
        self.line.set_data([], [])
        self.rg.set_data([], [])

    def update(self):
        e = self.element
        rg = e.rp + dot(e.Rp, e.Xc)
        self.rg.set_data(rg[self.x], rg[self.y])
        line = np.r_[[e.rp], [rg]]
        self.line.set_data(line[:, self.x], line[:, self.y])


class ModalBeamView(ElementView):
    element = DistalModalElementFromScratch

    def create_artists(self):
        self.shape, = self.axes.plot([], [], c='g', marker='o', lw=2, ms=1)
        self.local_axis, = self.axes.plot([], [], c='k', lw=2, alpha=0.3)
        return (self.shape, self.local_axis)

    def reset(self):
        self.shape.set_data([], [])
        self.local_axis.set_data([], [])

    def update(self):
        e = self.element
        X0, defl = self.element.local_deflections()
        shape = e.rp + array([dot(e.Rp, (X0[i]+defl[i]))
                              for i in range(X0.shape[0])])
        local_axis = np.r_[
            [e.rp],
            [e.rp + dot(e.Rp, X0[-1])],
            [shape[-1]]
        ]

        self.shape.set_data(shape[:, self.x], shape[:, self.y])
        self.local_axis.set_data(local_axis[:, self.x], local_axis[:, self.y])


class ModalElementView(ElementView):
    element = ModalElement

    def create_artists(self):
        self.shape, = self.axes.plot([], [], c='g', marker='o', lw=2, ms=1)
        self.local_axis, = self.axes.plot([], [], c='k', lw=2, alpha=0.3)
        return (self.shape, self.local_axis)

    def reset(self):
        self.shape.set_data([], [])
        self.local_axis.set_data([], [])

    def update(self):
        e = self.element
        shape = self.element.station_positions()
        local_axis = np.r_[
            [e.rp],
            [e.rp + dot(e.Rp, e.modes.X0[-1])],
            [shape[-1]]
        ]
        self.shape.set_data(shape[:, self.x], shape[:, self.y])
        self.local_axis.set_data(local_axis[:, self.x], local_axis[:, self.y])
