# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 11:09:08 2012

@author: Rick Lupton
"""

class Wind(object):
    def __init__(self, speed):
        self.speed = speed
    
    def get_wind_speed(self, time):
        """
        Get the three global components of wind speed at the specified time
        """
        return [self.speed, 0.0, 0.0]     
    
    
