# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:47:20 2012

@author: Rick Lupton
"""

import blade
import dynamics

dynamics.OPT_GRAVITY = False

tower = blade.Tower('/bladed/simple_tower/point_load/flextower_rigidblades_7modes.$pj')
rep,system,beams,att = tower.modal_analysis(4)

tower0 = blade.Tower('/bladed/simple_tower/point_load/tower_7modes_only.$pj')
modes = tower.modal_rep()
modes0 = tower0.modal_rep()