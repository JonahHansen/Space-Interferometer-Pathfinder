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
antisun_angle = np.radians(180)

dec_flag = 1
#Calculate required inclination from precession rate
def i_from_precession(rho):
    cosi = -2*rho*R_orb**3.5/(3*J2*R_e**2*np.sqrt(mu))
    return np.arccos(cosi)

#Desired rate of precession
precess_rate = np.radians(360)/(365.25*24*60*60)
#Inclination from precession
#inc_0 = i_from_precession(precess_rate)
inc_0 = np.radians(39)
#------------------------------------------------------------------------------------------
#Calculate orbit, in the geocentric (ECI) frame
ref = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, 0, 0)

#Number of orbits
n_orbits = int(365.25*24*60*60/ref.period)
#Number of phases in each orbit
n_phases = 40
#Total evaluation points
n_times = int(n_orbits*n_phases)
times = np.linspace(0,ref.period*n_orbits,n_times) #Create list of times

if dec_flag:
    n_ra = 1
    n_dec = 180
    ras = [np.radians(180)]
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

fig = plt.figure(1)
fig.clf()
ax1 = plt.subplot(1,1,1)
fig.subplots_adjust(bottom=0.2)
if dec_flag:
    plt.imshow(B[:,0,:].transpose(),aspect="auto",extent=[0,n_orbits,-90,90],cmap="inferno")
    ax1.set_ylabel(r"Declination $\delta$ $[\degree]$")
    plt.title("Observability over a year \n "+r"$i = %d\degree$, $\Omega = %d\degree$, $\alpha = %d\degree$, $\gamma= %d\degree$"%(round(np.degrees(inc_0)),round(np.degrees(Om_0)),round(np.degrees(ras[0])),round(np.degrees(antisun_angle))))

else:
    plt.imshow(B[:,0,:].transpose(),aspect="auto",extent=[0,n_orbits,0,360],cmap="inferno")
    plt.ylabel("Ra (degrees)")
    plt.title("Observability over a year \n "+r"$i = %d\degree$, $\Omega = %d\degree$, $\delta = 0\degree$, $\gamma= %d\degree$"%(round(np.degrees(inc_0)),round(np.degrees(Om_0)),round(np.degrees(antisun_angle))))

ax1.set_xlabel("Orbit number")
cbar = plt.colorbar()
cbar.set_label("Percentage viewable over an orbit")
plt.clim(0,100)

ax2 = ax1.twiny()


ax2.xaxis.set_ticks_position("bottom")
ax2.xaxis.set_label_position("bottom")

# Offset the twin axis below the host
ax2.spines["bottom"].set_position(("axes", -0.15))

# Turn on the frame for the twin axis, but then hide all
# but the bottom spine
ax2.set_frame_on(True)
ax2.patch.set_visible(False)

new_tick_locations = np.linspace(0,1,7)

def tick_function(X):
    V = 1/(1+X)
    return ["%.3f" % z for z in V]

ax2.spines["bottom"].set_visible(True)
ax2.set_xticks(new_tick_locations)
ax2.set_xticklabels(np.linspace(0,ref.period*n_orbits/60/60/24,7).astype(int))
ax2.set_xlabel(r"Time (Days)")

plt.savefig('Obs_om%d_inc%d_ra%d_as%d.svg'%(round(np.degrees(Om_0)),round(np.degrees(inc_0)),round(np.degrees(ras[0])),round(np.degrees(antisun_angle))), format='svg')

"""
if dec_flag:
    plt.savefig('Obs_om%d_inc%d_dec_as%d.svg'%(round(np.degrees(Om_0)),round(np.degrees(inc_0)),round(np.degrees(antisun_angle))), format='svg')
else:
    plt.savefig('Obs_om%d_inc%d_ra_as%d.svg'%(round(np.degrees(Om_0)),round(np.degrees(inc_0)),round(np.degrees(antisun_angle))), format='svg')
"""
