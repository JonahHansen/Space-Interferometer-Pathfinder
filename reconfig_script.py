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

#Stellar vector
ra = np.radians(float(sys.argv[1])) #90
dec = np.radians(float(sys.argv[2]))#-40

ra2 = np.radians(float(sys.argv[3]))
dec2 = np.radians(float(sys.argv[4]))

#The max distance to the other satellites in m
delta_r_max = 0.3e3

delta_r_max_mult = float(sys.argv[5])

#------------------------------------------------------------------------------------------
#Calculate reference orbit, in the geocentric (ECI) frame (See Orbit module)
ref1 = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)
ref2 = orbits.Reference_orbit(R_orb, delta_r_max_mult*delta_r_max, inc_0, Om_0, ra2, dec2)

reconfig.del_v_reconfigure(ref1,ref2)
