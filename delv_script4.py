import numpy as np
import astropy.constants as const
import modules.orbits as orbits
import sys
import delv_functions as delv_f
import matplotlib.pyplot as plt

alt = 500e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(40) #20
#Longitude of the Ascending Node
Om_0 = np.radians(90) #0

#Stellar vector
ra = np.radians(29) #90
dec = np.radians(45)#-40

#The max distance to the other satellites in m
delta_r_max = 0.3e3

#Calculate reference orbit, in the geocentric (ECI) frame (See Orbit module)
ref = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

#t_burn = float(sys.argv[1])
#zeta = float(sys.argv[2])

#Initial states of the satellites
c0 = orbits.init_chief(ref).state
d10 = orbits.init_deputy(ref,1).state
d20 = orbits.init_deputy(ref,2).state

n_orbits = 2
period = ref.periodK

for ix in range(4):

    print("Beginning orbit %s"%ix)

    t0 = ix*period
    t_final = (ix+0.5)*period

    c, d1, d2, tbank = delv_f.integration_fix2(ref, c0, d10, d20, t0, t_final)

    d1_f,d2_f = delv_f.reset_star_n_baseline(ref,c[-1],d1[-1],d2[-1],tbank[-1])

    delv_f.plotit(ix*2+1,ref,tbank,c,d1,d2)

    t02 = tbank[-1]
    t_end = (ix+1)*period
    n_burns = 2
    burn_times = [t02, t02 + 10*60, t_end - 10*60, t_end]

    c2, d12, d22, delv2, tbank2 = delv_f.recharge_fix(ref, c[-1], d1_f, d2_f, n_burns, burn_times)

    print("Delv for orbit %s: "%ix + str(np.sum(delv2,axis=0)))

    delv_f.plotit(ix*2+2,ref,tbank2,c2,d12,d22)

    c0 = c2[-1]
    d10 = d12[-1]
    d20 = d22[-1]

ix = 4
print("Beginning orbit %s"%ix)
t0 = ix*period
t_final = (ix+0.5)*period

c, d1, d2, tbank = delv_f.integration_fix2(ref, c0, d10, d20, t0, t_final)

d1_f,d2_f = delv_f.reset_star_n_baseline(ref,c[-1],d1[-1],d2[-1],tbank[-1])

delv_f.plotit(ix*2+1,ref,tbank,c,d1,d2)

#plt.show()
