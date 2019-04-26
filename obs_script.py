from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const
from orbits import ECI_orbit
from observability import check_obs
#import quaternions as qt

plt.ion()

alt = 500e3 #In m
R_e = const.R_earth.value  #In m
mu = const.GM_earth.value
J2 = 0.00108263

#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
#inc_0 = np.radians(20) #20
#Longitude of the Ascending Node
Om_0 = np.radians(0) #0

#Stellar vector
ra = np.radians(100) #90
dec = np.radians(-40)#-40

#The max distance to the other satellites in m
delta_r_max = 0.3e3

#List of perturbations: 1 = J2, 2 = Solar radiation, 3 = Drag. Leave empty list if no perturbations.
p_list = [1] #Currently just using J2

solar_angle = np.radians(60)

def i_from_precession(rho):
    cosi = 2*rho*R_orb**3.5/(3*J2*R_e**2*np.sqrt(mu))
    return np.arccos(cosi)
    
def ang_precession(i):
    return -3*np.pi*J2*R_e**2/(R_orb**2)*np.cos(i)

precession = np.radians(360)/(365.25*24*60*60)

inc_0 = i_from_precession(precession)

del_Om = ang_precession(inc_0)

#------------------------------------------------------------------------------------------
#Calculate orbit, in the geocentric (ECI) frame
ECI = ECI_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

Om = Om_0

n_orbits = int(365.25*24*60*60/ECI.period)
n_phases = int(ECI.period/60)
n_times = n_orbits*n_phases
times = np.linspace(0,ECI.period*n_orbits,n_times) #Create list of times

"""Initialise arrays"""
ECI_rc = np.zeros((n_times,6)) #Chief ECI position vector
ECI_rd1 = np.zeros((n_times,6)) #Deputy 1 ECI position vector
ECI_rd2 = np.zeros((n_times,6)) #Deputy 2 ECI position vector
obs = np.zeros(n_times)
u_v = np.zeros((n_times,2))

i = 0

for orbit in range(n_orbits):
    for phase in range(n_phases):
        t = times[orbit*n_phases + phase]
        ECI_rc[i] = ECI.chief_state(t)
        ECI_rd1[i] = ECI.deputy1_state(ECI_rc[i]) #Deputy 1 position
        ECI_rd2[i] = ECI.deputy2_state(ECI_rc[i]) #Deputy 2 position
        obs[i] = check_obs(t,ECI_rd1[i],ECI_rd2[i],ECI.s_hat,solar_angle)
        if obs[i]:
            u_v[i] = ECI.uv(ECI_rd1[i],ECI_rd2[i])
        i += 1
    print(i*100/n_times)
    Om += del_Om
    ECI = ECI_orbit(R_orb, delta_r_max, inc_0, Om, ra, dec)
    
plt.scatter(u_v[:,0],u_v[:,1],s=1)
plt.xlabel("u(m)")
plt.ylabel("v(m)")
plt.title("uv plane over a year")

percent = sum(obs)/len(obs)*100
print("Percentage viewable over a year: %.3f"%percent)
