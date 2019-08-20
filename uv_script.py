from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const
import modules.orbits as orbits
from modules.observability import check_obs

plt.ion()

#Altitude of satellite
alt = 500e3 #In m
#Radius of the Earth
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Earth's gravitational parameter
mu = const.GM_earth.value
#Earth J2 Term
J2 = 0.00108263

#Longitude of the Ascending Node
Om_0 = np.radians(0) #0

#Stellar vector
ra = np.radians(0) #90
dec = np.radians(45)#-40

#The max distance to the other satellites in m
delta_r_max = 0.3e3

#Angle within anti-sun axis
antisun_angle = np.radians(40)

#Calculate required inclination from precession rate
def i_from_precession(rho):
    cosi = 2*rho*R_orb**3.5/(3*J2*R_e**2*np.sqrt(mu))
    return np.arccos(cosi)

#Desired rate of precession
precess_rate = np.radians(360)/(365.25*24*60*60)
#Inclination from precession
inc_0 = i_from_precession(precess_rate)
inc_0 = np.radians(39)
#------------------------------------------------------------------------------------------
#Calculate orbit, in the geocentric (ECI) frame
ref = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

#Number of orbits
n_orbits = 365.25*24*60*60/ref.period
#Number of phases in each orbit
n_phases = ref.period/60/2
#Total evaluation points
n_times = int(n_orbits*n_phases)
times = np.linspace(0,ref.period*n_orbits,n_times) #Create list of times

"""Initialise arrays"""
obs = np.zeros(n_times) #Observable? array
u_v = np.zeros((n_times,2)) #uv point array

time_for_observing = 45 #Min
obs_num = int(n_phases/(ref.period/60/time_for_observing))

i = 0
j = 0
for t in times:
    pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(t)
    dep1 = orbits.init_deputy(ref,1,time=t) #Deputy 1 position
    dep2 = orbits.init_deputy(ref,2,time=t) #Deputy 2 position
    obs[i] = check_obs(t,ref.s_hat,pos_ref,antisun_angle,ref,False) #Check if observable
    if obs[i]:
        j += 1
        u_v[i] = ref.uv(dep1,dep2) #Find uv point if observable
    else:
        if j < obs_num:
            for k in range(j):
                obs[i-1-k] = 0
                u_v[i-1-k] = [0,0]
        j = 0
    i += 1
    print(i*100/n_times)

neg_uv = -u_v
uv = np.concatenate((u_v,neg_uv))
#plt.clf()
plt.scatter(uv[:,0],uv[:,1],s=1)
plt.xlabel("u(m)")
plt.ylabel("v(m)")
#plt.title("UV plane over a year, anti-sun angle = %s degrees"%round(np.degrees(antisun_angle)))
"""
antisun_angle = np.radians(30)

i = 0
j = 0
for t in times:
    pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(t)
    dep1 = orbits.init_deputy(ref,1,time=t) #Deputy 1 position
    dep2 = orbits.init_deputy(ref,2,time=t) #Deputy 2 position
    obs[i] = check_obs(t,ref.s_hat,pos_ref,antisun_angle,ref,False) #Check if observable
    if obs[i]:
        j += 1
        u_v[i] = ref.uv(dep1,dep2) #Find uv point if observable
    else:
        if j < obs_num:
            for k in range(j):
                obs[i-1-k] = 0
                u_v[i-1-k] = [0,0]
        j = 0
    i += 1
    print(i*100/n_times)

neg_uv = -u_v
uv2 = np.concatenate((u_v,neg_uv))
#plt.clf()
plt.scatter(uv2[:,0],uv2[:,1],s=1)
#plt.xlabel("u(m)")
#plt.ylabel("v(m)")
#plt.title("UV plane over a year, anti-sun angle = %s degrees"%round(np.degrees(antisun_angle)))

percent = sum(obs)/len(obs)*100
print("Percentage viewable over a year: %.3f"%percent)
"""
