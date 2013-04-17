
import numpy as np
from numpy import array, dot


class HingeView(object):
    def shape(self, hinge):
        # proximal values
        return [
            array([hinge.rd]),
            hinge.rd + np.r_[[dot(hinge.Rd, [1, 0, 0])],  [[0, 0, 0]],
                             [dot(hinge.Rd, [0, 1, 0])],  [[0, 0, 0]],
                             [dot(hinge.Rd, [0, 0, 1])]]
        ]

    shape_plot_options = [
        {'marker': 's', 'ms': 4, 'c': 'r'},
        {'c': 'k', 'lw': 0.5}
    ]


class PrismaticJointView(object):
    def shape(self, joint):
        # proximal values
        return [
            array([joint.rd]),
            joint.rd + np.r_[[dot(joint.Rd, [1, 0, 0])], [[0, 0, 0]],
                             [dot(joint.Rd, [0, 1, 0])], [[0, 0, 0]],
                             [dot(joint.Rd, [0, 0, 1])]]
        ]

    shape_plot_options = [
        {'marker': 's', 'ms': 4, 'c': 'r'},
        {'c': 'k', 'lw': 0.5}
    ]


class FreeJointView(object):
    def shape(self, joint):
        # proximal values
        return [
            array([joint.rd]),
            joint.rd + np.r_[[dot(joint.Rd, [1, 0, 0])], [[0, 0, 0]],
                             [dot(joint.Rd, [0, 1, 0])], [[0, 0, 0]],
                             [dot(joint.Rd, [0, 0, 1])]]
        ]

    shape_plot_options = [
        {'marker': 's', 'ms': 4, 'c': 'r'},
        {'c': 'k', 'lw': 0.5}
    ]


class RigidConnectionView(object):
    def shape(self, joint):
        return [
            np.r_[[joint.rp], [joint.rd]],
            joint.rd + np.r_[[dot(joint.Rd, [1, 0, 0])], [[0, 0, 0]],
                             [dot(joint.Rd, [0, 1, 0])], [[0, 0, 0]],
                             [dot(joint.Rd, [0, 0, 1])]]
        ]

    shape_plot_options = [
        {'c': 'm', 'lw': 1},
        {'c': 'k', 'lw': 0.5}
    ]


class RigidBodyView(object):
    def shape(self, body):
        return [
            np.r_[[body.rp], [body.rp + dot(body.Rp, body.Xc)]],
            np.r_[[body.rp+dot(body.Rp, body.Xc)]]
        ]

    shape_plot_options = [
        {'marker': 'x', 'ms': 4,  'c': 'y', 'mew': 2},
        {'marker': 'x', 'ms': 10, 'c': 'y', 'mew': 3},
    ]


def plot_chain(self, ax, **opts):
    for linedata,lineopts in zip(self.shape(), self.shape_plot_options):
        for k,v in opts.items():
            lineopts.setdefault(k, v) # set if not already specified
        ax.plot(linedata[:,0], linedata[:,1], linedata[:,2], **lineopts)
    for node_children in self.children:
        for child in node_children:
            child.plot_chain(ax, **opts)
