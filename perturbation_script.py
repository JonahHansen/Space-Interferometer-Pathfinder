from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits as orbits
from matplotlib.collections import LineCollection
from modules.Schweighart_J2_solved import propagate_spacecraft

plt.ion()

alt = 500e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(90) #20
#Longitude of the Ascending Node
Om_0 = np.radians(90) #0

#Stellar vector
ra = np.radians(0) #90
dec = np.radians(0)#-40

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

t0 = 2*np.pi/(8*ref.Sch_k)

#Initial reference orbit state
pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(t0)

HCW = 0

if HCW:
    precession = False
else:
    precession = True

#Initial states of the satellites
chief_0 = orbits.init_chief(ref,precession,time=t0).to_Curvy(precession=precession,ref_orbit=True)
deputy1_0 = orbits.init_deputy(ref,1,precession,time=t0).to_Curvy(precession=precession,ref_orbit=True)
deputy2_0 = orbits.init_deputy(ref,2,precession,time=t0).to_Curvy(precession=precession,ref_orbit=True)

chief_p_states = propagate_spacecraft(t0,chief_0.state,times,ref,HCW=HCW).transpose()
deputy1_p_states = propagate_spacecraft(t0,deputy1_0.state,times,ref,HCW=HCW).transpose()
deputy2_p_states = propagate_spacecraft(t0,deputy2_0.state,times,ref,HCW=HCW).transpose()

d1_rel_states = deputy1_p_states# - chief_p_states
d2_rel_states = deputy2_p_states# - chief_p_states

ECI_rc = np.zeros((len(times),3))
rel_p_dep1 = []
rel_p_dep2 = []

print("Integration Done")
for i in range(len(times)):
    rel_p_dep1.append(orbits.Curvy_Sat(d1_rel_states[i,:3],d1_rel_states[i,3:],times[i],ref))#.to_Baseline(precession=precession,ref_orbit=True))
    rel_p_dep2.append(orbits.Curvy_Sat(d2_rel_states[i,:3],d2_rel_states[i,3:],times[i],ref))#.to_Baseline(precession=precession,ref_orbit=True))
print("Classifying Done")


d1_pos = np.zeros((n_times,3)) #Deputy1 position in star direction
d2_pos = np.zeros((n_times,3)) #Deputy2 position in star direction

for ix in range(n_times):
    d1_pos[ix] = rel_p_dep1[ix].pos
    d2_pos[ix] = rel_p_dep2[ix].pos

plt.figure(1)
plt.clf()

plt.subplot(3,1,1)
plt.plot(times,d1_pos[:,0],'b-',label="Deputy1")
plt.plot(times,d2_pos[:,0],'r-',label="Deputy2")
plt.ylabel(r"$\rho$ Separation (m)")
plt.title("Separations in the LVLH Frame \n "+r"$i = %d\degree$, $\Omega = %d\degree$, $\alpha = %d\degree$, $\delta = %d\degree$"%(np.degrees(inc_0),np.degrees(Om_0),np.degrees(ra),np.degrees(dec)))

plt.legend()
plt.subplot(3,1,2)
plt.plot(times,d1_pos[:,1],'b-',label="Deputy1")
plt.plot(times,d2_pos[:,1],'r-',label="Deputy2")
plt.ylabel(r"$\xi$ Separation (m)")


plt.subplot(3,1,3)
plt.plot(times,d1_pos[:,2],'b-',label="Deputy1")
plt.plot(times,d2_pos[:,2],'r-',label="Deputy2")
plt.ylabel(r"$\eta$ Separation (m)")
plt.xlabel("Time (s)")

plt.savefig('SCH_90.svg', format='svg')
