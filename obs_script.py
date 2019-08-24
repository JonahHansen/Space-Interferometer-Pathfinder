from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const
import modules.orbits as orbits
from modules.observability import check_obs
import modules.quaternions as qt

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
Om_0 = np.radians(90) #0

#The max distance to the other satellites in m
delta_r_max = 0.3e3

#Angle within anti-sun axis
antisun_angle = np.radians(60)

dec_flag = 0
#Calculate required inclination from precession rate
def i_from_precession(rho):
    cosi = -2*rho*R_orb**3.5/(3*J2*R_e**2*np.sqrt(mu))
    return np.arccos(cosi)

#Desired rate of precession
precess_rate = np.radians(360)/(365.25*24*60*60)
#Inclination from precession
inc_0 = i_from_precession(precess_rate)
inc_0 = np.radians(39)
#------------------------------------------------------------------------------------------
#Calculate orbit, in the geocentric (ECI) frame
ref = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, 0, 0)

#Number of orbits
n_orbits = int(365.25*24*60*60/ref.period)
#Number of phases in each orbit
n_phases = 30
#Total evaluation points
n_times = int(n_orbits*n_phases)
times = np.linspace(0,ref.period*n_orbits,n_times) #Create list of times

if dec_flag:
    n_ra = 1
    n_dec = 180
    ras = np.linspace(0,np.radians(360),n_ra)
    decs = np.linspace(np.radians(-90),np.radians(90),n_dec)

else:
    n_ra = 180
    n_dec = 1
    ras = np.linspace(0,np.radians(360),n_ra)
    decs = np.array([0])

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
B = np.mean(np.reshape(obs,(int(obs.shape[0]/n_phases),n_phases,1,180)),1)*100

plt.clf()
if dec_flag:
    plt.imshow(B[:,0,:].transpose(),aspect="auto",extent=[0,n_orbits,-90,90],cmap="inferno")
    plt.ylabel("Dec (degrees)")
    plt.title("Observability over a year \n "+r"$i = %d\degree$, $\Omega = %d\degree$, $\alpha = 0\degree$, $\gamma= %d\degree$"%(round(np.degrees(inc_0)),round(np.degrees(Om_0)),round(np.degrees(antisun_angle))))

else:
    plt.imshow(B[:,0,:].transpose(),aspect="auto",extent=[0,n_orbits,0,360],cmap="inferno")
    plt.ylabel("Ra (degrees)")
    plt.title("Observability over a year \n "+r"$i = %d\degree$, $\Omega = %d\degree$, $\delta = 0\degree$, $\gamma= %d\degree$"%(round(np.degrees(inc_0)),round(np.degrees(Om_0)),round(np.degrees(antisun_angle))))

plt.xlabel("Orbit number")
cbar = plt.colorbar()
cbar.set_label("Percentage viewable over an orbit")
plt.clim(0,100)
if dec_flag:
    plt.savefig('Obs_om%d_inc%d_dec_as%d.svg'%(round(np.degrees(Om_0)),round(np.degrees(inc_0)),round(np.degrees(antisun_angle))), format='svg')
else:
    plt.savefig('Obs_om%d_inc%d_ra_as%d.svg'%(round(np.degrees(Om_0)),round(np.degrees(inc_0)),round(np.degrees(antisun_angle))), format='svg')
