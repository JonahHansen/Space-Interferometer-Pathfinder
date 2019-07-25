from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const
import modules.orbits as orbits
from modules.observability import check_obs
from multiprocessing import Pool
import json

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

#Orbital inclination
#inc_0 = np.radians(20) #20
#Longitude of the Ascending Node
Om_0 = np.radians(0) #0

#The max distance to the other satellites in m
delta_r_max = 0.3e3

#Calculate required inclination from precession rate
def i_from_precession(rho):
    cosi = 2*rho*R_orb**3.5/(3*J2*R_e**2*np.sqrt(mu))
    return np.arccos(cosi)
    
#-----------------------------------------------------------------------------------------
#Angle within anti-sun axis
antisun_angle = np.radians(40)

#Desired rate of precession
precess_rate = np.radians(360)/(365.25*24*60*60)
#Inclination from precession
inc_0 = i_from_precession(precess_rate)

#------------------------------------------------------------------------------------------
def worker(radec):
    
    (ra,dec) = radec
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
    
    time_for_observing = 45 #Min
    obs_num = int(n_phases/(ref.period/60/time_for_observing))
    
    i = 0
    j = 0
    for t in times:
        chief = orbits.init_chief(ref,t) #Include precession
        dep1 = orbits.init_deputy(ref,t,1) #Deputy 1 position
        dep2 = orbits.init_deputy(ref,t,2) #Deputy 2 position
        obs[i] = check_obs(t,dep1,dep2,antisun_angle,ref) #Check if observable
        if obs[i]:
            j += 1
        else:
            if j < obs_num:
                for k in range(j):
                    obs[i-1-k] = 0
            j = 0
        i += 1
        
    x = list(split(obs,12))
    y = [sum(a)/len(a)*100 for a in x]
    
    percent = sum(obs)/len(obs)*100
    output_dict = {"radec":radec, "percent":percent, "Calendar":y}
    
    return output_dict

inputs = [(np.radians(ra),np.radians(dec)) for ra in range(0,360,2) for dec in range(-90,90,2)]

p = Pool(processes=28)

result = p.map(worker,inputs)

with open('hopeful_H40.json', 'w') as f:  # writing JSON object
     json.dump(result, f)
