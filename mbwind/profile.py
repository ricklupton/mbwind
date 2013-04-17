
import pstats, cProfile

from dynamics import *

tower_height = 90.0
overhang = 0.8
r_root = 0.5
blade_length = 60.0

EIy = 1000e6
EIz = 1000e6

Ry = rotmat_y(-pi/2)
Rhb1 = rotmat_x(0 * 2*pi/3)
Rhb2 = rotmat_x(1 * 2*pi/3)
Rhb3 = rotmat_x(2 * 2*pi/3)

foundation = Hinge('foundation', [0,1,0], rotmat_y(-pi/2))
tower = EulerBeam('tower', tower_height, 3000, 100e6, 300e6, 300e6, 200e6)
nacelle = RigidConnection('nacelle', [0,0,-overhang], rotmat_y(80*pi/180))
bearing = Hinge('bearing', [1,0,0]) 
hb1 = RigidConnection('hb1', np.dot(Rhb1,[0,0,1]), np.dot(Rhb1,Ry))
hb2 = RigidConnection('hb2', np.dot(Rhb2,[0,0,1]), np.dot(Rhb2,Ry))
hb3 = RigidConnection('hb3', np.dot(Rhb3,[0,0,1]), np.dot(Rhb3,Ry))
b1 = EulerBeam('b1',blade_length, 250, 1000e6, EIy, EIz, 200e6)
b2 = EulerBeam('b2',blade_length, 250, 1000e6, EIy, EIz, 200e6)
b3 = EulerBeam('b3',blade_length, 250, 1000e6, EIy, EIz, 200e6)

tower.damping = 0.05

foundation.add_leaf(tower)
tower.add_leaf(nacelle)
nacelle.add_leaf(bearing)
bearing.add_leaf(hb1)
bearing.add_leaf(hb2)
bearing.add_leaf(hb3)
hb1.add_leaf(b1)
hb2.add_leaf(b2)
hb3.add_leaf(b3)
system = System(foundation)

cProfile.runctx("for i in range(100): system.update()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

