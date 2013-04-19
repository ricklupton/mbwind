import numpy as np
from numpy import array, zeros
import re
from .modes import ModalRepresentation
from pybladed.model import Model, ParameterNotFound


class BladedModesReader(object):
    def __init__(self, filename):
        self.data = None
        self.load(filename)

    def load(self, filename):
        self.filename = filename

        # read header file and find RMODE module
        with open(filename, 'rU') as f:
            s = f.read()
        model = Model(s)

        try:
            model.get_param('MULTIBODY')
        except ParameterNotFound:
            raise NotImplemented('Only Bladed versions >= v4 are supported')

        # Blade station radii (values duplicated for split station)
        self.radii = model.get_param(('BGEOMMB', 'RJ')).split(',')
        self.radii = array(map(float, self.radii[::2]))

        # Mass axis coordinates
        #mass_axis = model.get_param(('BGEOMMB','')).split(',')

        try:
            module = model.get_module('RMODE')
            module = re.search(r'''
                    NBLADE[ \t]+(\d+)\n
                    (?:TYPE[ \t]+(.*)\n
                    FREQ[ \t]+(.*)\n
                    DAMP[ \t]+(.*)\n
                    ((?:MD.*\n)+|CRYPT))?
                ''', module, re.VERBOSE).groups()
        except (ParameterNotFound, AttributeError):
            raise Exception("Couldn't read RMODE module from '%s'" % filename)

        nmodes = int(module[0])
        self.data = np.zeros((6, nmodes, len(self.radii)))
        if nmodes > 0:
            self.types = module[1].split()
            self.freqs = array(map(float, module[2].split()))
            self.damping = array(map(float, module[3].split()))
            rest = module[4]

            if rest == 'CRYPT':
                raise NotImplemented("Cannot read encrypted modes")

            for i, line in enumerate(rest.splitlines()):
                imode = int(i/6)
                icomp = i % 6
                assert line.startswith('MD%02d%d' % (1+imode, 1+icomp))
                d = [float(xx.strip(',')) for xx in line[5:].split()]
                self.data[icomp, imode, :] = d
        else:
            raise Exception('No modes found in "%s"' % filename)

        # extract channel names
        self.axnames = ['Component', 'Mode', 'Radius']
        self.axticks = [
            ['x', 'y', 'z', 'Rx', 'Ry', 'Rz'],
            ['%d(%s)' % xx for xx in zip(range(1, 1+nmodes), self.types)],
            ['%g' % xx for xx in self.radii],
        ]
        self.axvals = [
            range(6),
            range(nmodes),
            self.radii,
        ]

        # Other data (ignore split stations)
        self.density = model.get_param(('BMASSMB', 'MASS')).split(',')
        self.density = array(map(float, self.density[::2]))

        # Stiffness
        self.EI_flap = model.get_param(('BSTIFFMB', 'EIFLAP')).split(',')
        self.EI_flap = array(map(float, self.EI_flap)).reshape((-1, 2))
        self.EI_edge = model.get_param(('BSTIFFMB', 'EIEDGE')).split(',')
        self.EI_edge = array(map(float, self.EI_edge)).reshape((-1, 2))


def load_modes_from_Bladed(filename):
    """Load modal data from Bladed .prj or .$pj file."""
    bmf = BladedModesReader(filename)
    shapes_rotations = bmf.data.transpose(2, 0, 1)
    permute_axes = (2, 0, 1, 5, 3, 4)  # z -> x, x -> y, y -> z
    shapes_rotations = shapes_rotations[:, permute_axes, :]
    kgyr = np.zeros_like(bmf.radii)
    #mass_axis = bmf.mass_axis
    rep = ModalRepresentation(bmf.radii, shapes_rotations[:, :3, :],
                              shapes_rotations[:, 3:, :], bmf.density,
                              bmf.freqs, section_inertia=kgyr,
                              damping=bmf.damping)
    # , mass_axis=mass_axis)
    return rep


def convert_Bladed_attachment_modes(mode_names, freqs, damping, shapes,
                                    rotations):
    """
    Convert attachment modes from force -> displacement, and make sure
    there are always 6 of them. Normal modes unchanged.
    """

    Na = sum(1 for name in mode_names if 'attachment' in name)
    Nn = sum(1 for name in mode_names if 'normal' in name)
    assert Na + Nn == len(freqs)
    Pall = np.r_['1', shapes, rotations]

    # attachment (a) and normal (n)
    Paf = Pall[:, :, :Na]
    Pn = Pall[:, :, Na:]

    # add extension and torsion attachment modes
    padding = zeros((Paf.shape[0], Paf.shape[1], 1))
    Paf = np.r_['2', padding, Paf[:, :, :2], padding, Paf[:, :, 2:]]
    Paf[-1, 0, 0] = 1  # dummy extension mode
    Paf[-1, 3, 3] = 1  # dummy torsion mode

    # displacements at tip
    Paf_tip = Paf[-1]
    Paf_tip_inv = np.linalg.inv(Paf_tip)

    # transform attachment modes from forces to displacements
    Pa = np.einsum('hip,pj->hij', Paf, Paf_tip_inv)
    Pa[:, 0, 0] = 0  # dummy extension mode
    Pa[:, 3, 3] = 0  # dummy torsion mode

    # all mode shapes again
    P = np.r_['2', Pa, Pn]
    new_shapes = P[:, :3, :]
    new_rotations = P[:, 3:, :]
    new_freqs = np.r_[np.nan, freqs[:2], np.nan, freqs[2:]]
    new_damping = np.r_[np.nan, damping[:2], np.nan, damping[2:]]
    new_mode_names = (["Dummy extension mode"] + mode_names[:2] +
                      ["Dummy torsion mode"] + mode_names[2:])

    return new_mode_names, new_freqs, new_damping, new_shapes, new_rotations
