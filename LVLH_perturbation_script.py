""" Script to plot the unperturbed and J2 perturbed motion, calculated analytically in the LVLH frame """
""" Takes in inc, Om, ra, dec from the command line """

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits as orbits
from matplotlib.collections import LineCollection
from modules.Analytical_LVLH_motion import propagate_spacecraft
import sys

plt.ion()

alt = 500e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(float(sys.argv[1])) #20
#Longitude of the Ascending Node
Om_0 = np.radians(float(sys.argv[2])) #0

#Stellar vector
ra = np.radians(float(sys.argv[3])) #90
dec = np.radians(float(sys.argv[4]))#-40

#The max distance to the other satellites in m
delta_r_max = 0.3e3

#------------------------------------------------------------------------------------------
#Calculate reference orbit, in the geocentric (ECI) frame (See Orbit module)
ref = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

#Number of orbits
n_orbits = 10
#Number of phases in each orbit
n_phases = 100
#Total evaluation points
n_times = int(n_orbits*n_phases)
times = np.linspace(0,ref.period*n_orbits,n_times) #Create list of times

#Initial time to offset secular motion
t0 = 2*np.pi/(8*ref.Sch_k)

#Initial reference orbit state
pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(t0)

#Set flag if plotting unperturbed (HCW) motion. Otherwise, will use Schweighart.
HCW = 1

if HCW:
    precession = False
else:
    precession = True

#Initial states of the satellites in curvilinear frame
chief_0 = orbits.init_chief(ref,precession,time=t0).to_Curvy(precession=precession,ref_orbit=True)
deputy1_0 = orbits.init_deputy(ref,1,precession,time=t0).to_Curvy(precession=precession,ref_orbit=True)
deputy2_0 = orbits.init_deputy(ref,2,precession,time=t0).to_Curvy(precession=precession,ref_orbit=True)

#Propagate the spacecraft (see module for more details) into a list of states
chief_p_states = propagate_spacecraft(t0,chief_0.state,times,ref,HCW=HCW).transpose()
deputy1_p_states = propagate_spacecraft(t0,deputy1_0.state,times,ref,HCW=HCW).transpose()
deputy2_p_states = propagate_spacecraft(t0,deputy2_0.state,times,ref,HCW=HCW).transpose()

#Plot absolute or relative motion? (Doesn't matter for HCW, as chief is 0)
d1_rel_states = deputy1_p_states #- chief_p_states
d2_rel_states = deputy2_p_states #- chief_p_states

rel_p_dep1 = []
rel_p_dep2 = []

print("Integration Done")

#Change to satellite class and retrieve position
for i in range(len(times)):
    rel_p_dep1.append(orbits.Curvy_Sat(d1_rel_states[i,:3],d1_rel_states[i,3:],times[i],ref)).pos
    rel_p_dep2.append(orbits.Curvy_Sat(d2_rel_states[i,:3],d2_rel_states[i,3:],times[i],ref)).pos


#Change to minutes
times = times/60

#Plot the motion (edit based on requirements)
plt.figure(1)
plt.clf()
plt.tight_layout()
#plt.ticklabel_format(style='sci', axis='y', scilimits=(-2,2))
ax = plt.subplot(3,1,1)
ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='y')
plt.plot(times,rel_p_dep1[:,0],'b-',label="Deputy1")
plt.plot(times,rel_p_dep2[:,0],'r-',label="Deputy2")
plt.ylim(-1e-5,1e-5)
plt.ylabel(r"$\rho$ Separation (m)")
#plt.ylabel(r"$\rho$ (m)")
plt.title("Separations in the LVLH Frame \n "+r"$i = %d\degree$, $\Omega = %d\degree$, $\alpha = %d\degree$, $\delta = %d\degree$"%(np.degrees(inc_0),np.degrees(Om_0),np.degrees(ra),np.degrees(dec)))

plt.legend()
plt.subplot(3,1,2)
plt.plot(times,rel_p_dep1[:,1],'b-',label="Deputy1")
plt.plot(times,rel_p_dep2[:,1],'r-',label="Deputy2")
plt.ylabel(r"$\xi$ Separation (m)")
#plt.ylabel(r"$\xi$ (m)")

ax = plt.subplot(3,1,3)
ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='y')
plt.plot(times,rel_p_dep1[:,2],'b-',label="Deputy1")
plt.plot(times,rel_p_dep2[:,2],'r-',label="Deputy2")
plt.ylim(-1e-5,1e-5)
plt.ylabel(r"$\eta$ Separation (m)")
#plt.ylabel(r"$\eta$ (m)")
plt.xlabel("Time (min)")

#Save figure
plt.savefig('HCW_0.svg', format='svg')
