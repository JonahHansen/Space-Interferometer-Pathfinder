from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const
import modules.orbits as orbits
from modules.observability import check_obs
import modules.quaternions as qt

plt.ion()

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

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
Om_0 = np.radians(90) #0

#The max distance to the other satellites in m
delta_r_max = 0.3e3

#Angle within anti-sun axis
antisun_angle = np.radians(180)

#Calculate required inclination from precession rate
def i_from_precession(rho):
    cosi = -2*rho*R_orb**3.5/(3*J2*R_e**2*np.sqrt(mu))
    return np.arccos(cosi)

#Desired rate of precession
precess_rate = np.radians(360)/(365.25*24*60*60)
#Inclination from precession
inc_0 = i_from_precession(precess_rate)
#inc_0 = np.radians(39)
#------------------------------------------------------------------------------------------
#Calculate orbit, in the geocentric (ECI) frame
ref = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, 0, 0)

#Number of orbits
n_orbits = int(365.25*24*60*60/ref.period)
#Number of phases in each orbit
n_phases = 60
#Total evaluation points
n_times = int(n_orbits*n_phases)
times = np.linspace(0,ref.period*n_orbits,n_times) #Create list of times

n_ra = 360
n_dec = 180
ras = np.linspace(0,np.radians(360),n_ra)
decs = np.linspace(np.radians(-90),np.radians(90),n_dec)

ra,dec = np.meshgrid(ras,decs)

s_hats = np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)]).transpose()

"""Initialise arrays"""
obs = np.zeros((n_times,n_ra,n_dec)) #Observable? array

time_for_observing = 45 #Min
obs_num = int(n_phases/(ref.period/60/time_for_observing))

i = 0
j = np.zeros((n_ra,n_dec))

""" Initialise a deputy at a given time t from the reference orbit """
""" the n variable is for the number of the deputy (i.e 1 or 2) """

for t in times:
    pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(t)
    obs[i] = check_obs(t,s_hats,pos_ref,antisun_angle,ref) #Check if observable
    """
        for ix in range(n_ra):
            for iy in range(n_dec):
                if obs[i,ix,iy]:
                    j[ix,iy] += 1
                else:
                    if j[ix,iy] < obs_num:
                        for k in range(int(j[ix,iy])):
                            obs[i-1-k,ix,iy] = 0
                    j[ix,iy] = 0
    """
    i += 1
    print(i*100/n_times)

x = list(split(obs,12))
y = [sum(a)/len(a)*100 for a in x]

percent = sum(obs)/len(obs)*100
output_dict = {"radec":radec, "percent":percent, "Calendar":y}

with open('pkfire.json', 'w') as f:  # writing JSON object
     json.dump(result, f)
