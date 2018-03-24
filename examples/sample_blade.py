import numpy as np
import yaml
from beamfe import BeamFE, interleave


def sample_blade_data():
    # Load blade data
    with open('Bladed_models/parked_blade_nrel.yaml', 'r') as f:
        blade = yaml.safe_load(f)
    return blade


def blade_fe(blade, root_length=0.0, spin=0.0):
    # positive blade twist is geometrically a negative x-rotation
    twist = -np.array(blade['twist'])
    x = blade['radii']
    qn = interleave(root_length + np.array(x), 6)
    N = BeamFE.centrifugal_force_distribution(qn, blade['density'])
    fe = BeamFE(x, blade['density'], blade['EA'],
                blade['EIyy'], blade['EIzz'], twist=twist,
                axial_force=N * spin**2)
    fe.set_boundary_conditions('C', 'F')
    return fe


def sample_tower_data():
    # Load blade data
    with open('Bladed_models/simple_tower.yaml', 'r') as f:
        tower = yaml.safe_load(f)
    return tower


def tower_fe(tower):
    x = tower['radii']
    fe = BeamFE(x, tower['density'], tower['EA'],
                tower['EIyy'], tower['EIzz'])
    fe.set_dofs([False, True, True, False, True, True])
    fe.set_boundary_conditions('C', 'C')
    return fe
