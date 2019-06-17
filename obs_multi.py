from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const
from modules.orbits import ECI_orbit, Chief, init_deputy
from modules.observability import check_obs
from multiprocessing import Pool
import json

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

#Orbital inclination
#inc_0 = np.radians(20) #20
#Longitude of the Ascending Node
Om_0 = np.radians(0) #0

#The max distance to the other satellites in m
delta_r_max = 0.3e3

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

#Angle within anti-sun axis
solar_angle = np.radians(60)

#Calculate required inclination from precession rate
def i_from_precession(rho):
    cosi = 2*rho*R_orb**3.5/(3*J2*R_e**2*np.sqrt(mu))
    return np.arccos(cosi)

#Calculate the amount of precession in an orbit for a given inclination
def ang_precession(i):
    return -3*np.pi*J2*R_e**2/(R_orb**2)*np.cos(i)

#Desired rate of precession
precess_rate = np.radians(360)/(365.25*24*60*60)
#Inclination from precession
inc_0 = i_from_precession(precess_rate)
#Precession change per orbit
del_Om = ang_precession(inc_0)

#------------------------------------------------------------------------------------------
def worker(radec):
    #Stellar vector
    (ra,dec) = radec
    
    #Calculate orbit, in the geocentric (ECI) frame
    ECI = ECI_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)
    
    Om = Om_0
    
    #Number of orbits
    n_orbits = int(365.25*24*60*60/ECI.period)
    #Number of phases in each orbit
    n_phases = int(ECI.period/60)
    #Total evaluation points
    n_times = n_orbits*n_phases
    times = np.linspace(0,ECI.period*n_orbits,n_times) #Create list of times
    
    """Initialise arrays"""
    obs = np.zeros(n_times) #Observable? array
    #u_v = np.zeros((n_times,2)) #uv point array
    
    
    time_for_observing = 45 #Min
    obs_num = int(n_phases/(ECI.period/60/time_for_observing))

    i = 0
    j = 0
    
    for orbit in range(n_orbits):
        for phase in range(n_phases):
            t = times[orbit*n_phases + phase]
            ECI_rc = Chief(ECI,t)
            ECI_rd1 = init_deputy(ECI,ECI_rc,1) #Deputy 1 position
            ECI_rd2 = init_deputy(ECI,ECI_rc,2) #Deputy 2 position
            obs[i] = check_obs(t,ECI_rd1,ECI_rd2,ECI.R_orb,ECI.s_hat,solar_angle) #Check if observable
            if obs[i]:
                j += 1
            else:
                if j < obs_num:
                    for k in range(j):
                        obs[i-1-k] = 0
                j = 0
            i += 1
        #print(i*100/n_times)
        
        #Precess the orbit
        Om += del_Om
        ECI = ECI_orbit(R_orb, delta_r_max, inc_0, Om, ra, dec)
    
    x = list(split(obs,12))
    y = [sum(a)/len(a)*100 for a in x]
    
    percent = sum(obs)/len(obs)*100
    output_dict = {"radec":radec, "percent":percent, "Calendar":y}
    
    return output_dict

inputs = [(np.radians(ra),np.radians(dec)) for ra in range(0,360,2) for dec in range(-90,90,2)]

p = Pool(processes=28)

result = p.map(worker,inputs)

with open('bigboi_.json', 'w') as f:  # writing JSON object
     json.dump(result, f)
