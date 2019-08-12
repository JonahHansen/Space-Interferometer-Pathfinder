from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits as orbits
from matplotlib.collections import LineCollection
from modules.Schweighart_J2_solved_clean import propagate_spacecraft

plt.ion()

alt = 500e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(64) #20
#Longitude of the Ascending Node
Om_0 = np.radians(0) #0

#Stellar vector
ra = np.radians(0) #90
dec = np.radians(0)#-40

#The max distance to the other satellites in m
delta_r_max = 0.3e3


def del_v_func(c,d1,d2,t,pt,ref):
    sat0 = orbits.Curvy_Sat(c[:3],c[3:],t,ref).to_Baseline()
    sat1 = orbits.Curvy_Sat(d1[:3],d1[3:],t,ref).to_Baseline()
    sat2 = orbits.Curvy_Sat(d2[:3],d2[3:],t,ref).to_Baseline()
    
    dsat1 = sat1.state-sat0.state
    dsat2 = sat2.state-sat0.state
    
    delvs1 = np.zeros(3)
    delvs2 = np.zeros(3)
    delvb = np.zeros(3)
    
    delvs1[2] = -dsat1[2]/(t-pt)
    delvs2[2] = -dsat2[2]/(t-pt)
    
    #Calculate position vector to the midpoint of the two deputies
    del_b = dsat1[0:3] - dsat2[0:3] #Separation vector
    del_b_half = 0.5*del_b #Midpoint
    m0 = dsat1[0:3] + del_b_half #Midpoint from centre
    m0[2] = 0
    
    delvb = m0/(t-pt)
    delvb = np.array([0,0,0])
    
    delv = np.array([np.linalg.norm(delvs1),np.linalg.norm(delvs2),np.linalg.norm(delvb)])
    
    sat0.vel += delvb
    sat1.vel += delvs1
    sat2.vel += delvs2
    
    new_sat0 = sat0.to_Curvy().state
    new_sat1 = sat1.to_Curvy().state
    new_sat2 = sat2.to_Curvy().state
    
    return delv,new_sat0,new_sat1,new_sat2
    

#------------------------------------------------------------------------------------------
#Calculate reference orbit, in the geocentric (ECI) frame (See Orbit module)
ref = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

#Number of orbits
n_orbits = 0.5
#Number of phases in each orbit
n_phases = 500
#Total evaluation points
n_times = int(n_orbits*n_phases)
times = np.linspace(0,ref.period*n_orbits,n_times) #Create list of times

#Initial states of the satellites
chief_0 = orbits.init_chief(ref).to_LVLH().state
deputy1_0 = orbits.init_deputy(ref,1).to_LVLH().state
deputy2_0 = orbits.init_deputy(ref,2).to_LVLH().state

chief_states = np.array([chief_0])
deputy1_states = np.array([deputy1_0])
deputy2_states = np.array([deputy2_0])
delv_bank = []

t_burn = 1
t0 = 0.1
t_bank = np.array([0])

while t0 < times[-1]:
    burn_pt = t0 + t_burn
    ts = np.linspace(t0,burn_pt,t_burn) #Every 0.1s
    t_bank = np.append(t_bank,ts)
    chief_states = np.append(chief_states,propagate_spacecraft(t0,chief_0,ts,ref).transpose(),axis=0)
    deputy1_states = np.append(deputy1_states,propagate_spacecraft(t0,deputy1_0,ts,ref).transpose(),axis=0)
    deputy2_states = np.append(deputy2_states,propagate_spacecraft(t0,deputy2_0,ts,ref).transpose(),axis=0)
    
    delv,new_c,new_d1,new_d2 = del_v_func(chief_states[-1],deputy1_states[-1],deputy2_states[-1],burn_pt,t0,ref)
    
    delv_bank.append(delv)
    chief_0 = new_c
    deputy1_0 = new_d1
    deputy2_0 = new_d2
    t0 = burn_pt

d1_rel_states = deputy1_states - np.array(chief_states)
d2_rel_states = np.array(deputy2_states) - np.array(chief_states)

rel_p_dep1 = []
rel_p_dep2 = []

print("Integration Done")
for i in range(len(times)):
    pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(t_bank[i],True)
    rel_p_dep1.append(orbits.LVLH_Sat(d1_rel_states[i,:3],d1_rel_states[i,3:],t_bank[i],ref).to_Baseline())
    rel_p_dep2.append(orbits.LVLH_Sat(d2_rel_states[i,:3],d2_rel_states[i,3:],t_bank[i],ref).to_Baseline())
print("Classifying Done")

#--------------------------------------------------------------------------------------------- #
#Separations and accelerations
baseline_sep = np.zeros(n_times) #Separation along the baseline
s_hat_drd1 = np.zeros(n_times) #Deputy1 position in star direction
s_hat_drd2 = np.zeros(n_times) #Deputy2 position in star direction
b_hat_drd1 = np.zeros(n_times) #Deputy1 position in baseline direction
b_hat_drd2 = np.zeros(n_times) #Deputy2 position in baseline direction
s_hat_sep = np.zeros(n_times) #Separation along the baseline
total_sep = np.zeros(n_times) #Total separation

for ix in range(n_times):
    #Baseline separations is simply the difference between the positions of the two deputies
    baseline_sep[ix] = np.linalg.norm(rel_p_dep2[ix].pos) - np.linalg.norm(rel_p_dep1[ix].pos)

    #Component of perturbed orbit in star direction
    s_hat_drd1[ix] = rel_p_dep1[ix].pos[2]
    s_hat_drd2[ix] = rel_p_dep2[ix].pos[2]

    #Component of perturbed orbit in baseline direction
    b_hat_drd1[ix] = rel_p_dep1[ix].pos[0]
    b_hat_drd2[ix] = rel_p_dep2[ix].pos[0]

    #Separation of the two deputies in the star direction
    s_hat_sep[ix] = s_hat_drd1[ix] - s_hat_drd2[ix]
    #baseline_sep[ix] = b_hat_drd1[ix] + b_hat_drd2[ix]
    #Sum of the separation along the star direction and the baseline direction
    #total_sep[ix] = baseline_sep[ix] + s_hat_sep[ix]

print(np.sum(np.array(delv_bank),axis=0))
print(np.max(np.abs(s_hat_sep)))

# ---------------------------------------------------------------------- #
### PLOTTING STUFF ###

#Plot separation along the star direction
plt.figure(3)
plt.clf()
#plt.plot(times,s_hat_drd1,"b-",label="SCHWEIGHART Deputy 1, s direction")
#plt.plot(times,s_hat_drd2,"g-",label="SCHWEIGHART Deputy 2, s direction")
#plt.plot(times,s_hat_sep,"r-",label="SCHWEIGHART Separation, s direction")
plt.plot(times,s_hat_drd1,"b-",label="Deputy 1, s direction")
plt.plot(times,s_hat_drd2,"g-",label="Deputy 2, s direction")
plt.plot(times,s_hat_sep,"r-",label="Separation, s direction")
plt.xlabel("Times(s)")
plt.ylabel("Separation(m)")
plt.title('Separations against time due to perturbations')
plt.legend()

#Plot separation along the baseline direction
plt.figure(4)
plt.clf()
plt.plot(times,baseline_sep,"y-",label="Separation, baseline direction")
#plt.plot(times,total_sep,"c-",label="Total direction")
plt.xlabel("Times(s)")
plt.ylabel("Separation(m)")
plt.title('Separations against time due to perturbations')
plt.legend()

#Plot separation in the baseline frame
plt.figure(5)
plt.clf()

points1 = np.array([b_hat_drd1, s_hat_drd1]).T.reshape(-1, 1, 2)
points2 = np.array([b_hat_drd2, s_hat_drd2]).T.reshape(-1, 1, 2)
segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
norm = plt.Normalize(times.min(), times.max())
ax = plt.gca()
lc1 = LineCollection(segments1, cmap='YlOrRd', norm=norm)
lc1.set_array(times)
lc1.set_linewidth(2)
ax.add_collection(lc1)
lc2 = LineCollection(segments2, cmap='YlGnBu', norm=norm)
lc2.set_array(times)
lc2.set_linewidth(2)
ax.add_collection(lc2)
space_f = 1.2
plt.xlim(np.min(space_f*np.minimum(b_hat_drd1,b_hat_drd2)), np.max(space_f*np.maximum(b_hat_drd1,b_hat_drd2)))
plt.ylim(np.min(space_f*np.minimum(s_hat_drd1,s_hat_drd2)), np.max(space_f*np.maximum(s_hat_drd1,s_hat_drd2)))
plt.xlabel("Baseline direction (m)")
plt.ylabel("Star direction (m)")
plt.title("Position of deputies due to \n perturbations in Baseline frame")

cbar = plt.colorbar(lc1)
plt.colorbar(lc2)
#cbar.set_label('Time (Schweighart) (s)', rotation=270, labelpad = 15)
cbar.set_label('Time (s)', rotation=270, labelpad = 15)
