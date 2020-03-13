""" Script to calculate the amount of Delta v required to reconfigure orbits """
"""Takes in ra1, dec1, ra2, dec2, delta_r_max_mult in the command line """

import numpy as np
import modules.orbits as orbits
import sys
import astropy.constants as const
import modules.reconfiguration as reconfig


alt = 500e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(0) #20
#Longitude of the Ascending Node
Om_0 = np.radians(0) #0

#Original stellar vector
ra = np.radians(float(sys.argv[1])) #90
dec = np.radians(float(sys.argv[2]))#-40

#New stellar vector
ra2 = np.radians(float(sys.argv[3]))
dec2 = np.radians(float(sys.argv[4]))

#The max distance to the other satellites in m
delta_r_max = 0.3e3

#Change to baseline (as a factor of the old one)
delta_r_max_mult = float(sys.argv[5])

#------------------------------------------------------------------------------------------
#Calculate reference orbits, in the geocentric (ECI) frame (See Orbit module) for both configurations
ref1 = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)
ref2 = orbits.Reference_orbit(R_orb, delta_r_max_mult*delta_r_max, inc_0, Om_0, ra2, dec2)

#Calculate delta v
reconfig.del_v_reconfigure(ref1,ref2)
