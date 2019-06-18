from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits as orbits
from matplotlib.collections import LineCollection
from modules.Schweighart_J2 import J2_pet, J2_rel_pet

plt.ion()

alt = 500e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(80) #20
#Longitude of the Ascending Node
Om_0 = np.radians(0) #0

#Stellar vector
ra = np.radians(90) #90
dec = np.radians(-40)#-40

#The max distance to the other satellites in m
delta_r_max = 0.3e3

#List of perturbations: 1 = J2, 2 = Solar radiation, 3 = Drag. Leave empty list if no perturbations.
p_list = [1] #Currently just using J2

#------------------------------------------------------------------------------------------
#Calculate reference orbit, in the geocentric (ECI) frame (See Orbit module)
ref = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

#Number of orbits
n_orbits = 0.5
#Number of phases in each orbit
n_phases = 1000
#Total evaluation points
n_times = int(n_orbits*n_phases)
times = np.linspace(0,ref.period*n_orbits,n_times) #Create list of times

pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(0,True)
chief_0 = orbits.init_chief(ref,0,True).to_LVLH(pos_ref,LVLH)
deputy1_0 = orbits.init_deputy(ref,0,1,True).to_LVLH(pos_ref,LVLH)
deputy2_0 = orbits.init_deputy(ref,0,2,True).to_LVLH(pos_ref,LVLH)

#Equations of motion
J2_func0 = J2_pet(chief_0,ref)
J2_func1 = J2_pet(deputy1_0,ref)
J2_func2 = J2_pet(deputy2_0,ref)

J2_rel_func1 = J2_rel_pet(deputy1_0,ref,1)
J2_rel_func2 = J2_rel_pet(deputy2_0,ref,2)

#Tolerance and steps required for the integrator
rtol = 1e-9
atol = 1e-18
step = 10

#Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
X_d0 = solve_ivp(J2_func0, [times[0],times[-1]], chief_0.state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X_d0.success:
    raise Exception("Integration failed!!!!")

X_d1 = solve_ivp(J2_func1, [times[0],times[-1]], deputy1_0.state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X_d1.success:
    raise Exception("Integration failed!!!!")

X_d2 = solve_ivp(J2_func2, [times[0],times[-1]], deputy2_0.state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
if not X_d2.success:
    raise Exception("Integration failed!!!!")

X_rel_d1 = solve_ivp(J2_rel_func1, [times[0],times[-1]], deputy1_0.state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X_rel_d1.success:
    raise Exception("Integration failed!!!!")

X_rel_d2 = solve_ivp(J2_rel_func2, [times[0],times[-1]], deputy2_0.state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
if not X_rel_d2.success:
    raise Exception("Integration failed!!!!")

#Peturbed orbits
pert_chief = np.transpose(X_d0.y)
pert_deputy1 = np.transpose(X_d1.y)
pert_deputy2 = np.transpose(X_d2.y)

pert_rel_deputy1 = np.transpose(X_rel_d1.y)
pert_rel_deputy2 = np.transpose(X_rel_d2.y)

man_rel_dep1 = pert_deputy1 - pert_chief
man_rel_dep2 = pert_deputy2 - pert_chief

p_chief = []
p_dep1 = []
p_dep2 = []
rel_p_dep1 = []
rel_p_dep2 = []

print("Integration Done")
for i in range(len(pert_chief)):
    pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(times[i],True)
    #p_chief.append(orbits.LVLH_Sat(pert_chief[i,:3],pert_chief[i,3:],times[i],ref).to_Baseline(LVLH,Base))
    #p_dep1.append(orbits.LVLH_Sat(pert_deputy1[i,:3],pert_deputy1[i,3:],times[i],ref).to_Baseline(LVLH,Base))
    #p_dep2.append(orbits.LVLH_Sat(pert_deputy2[i,:3],pert_deputy2[i,3:],times[i],ref).to_Baseline(LVLH,Base))
    rel_p_dep1.append(orbits.LVLH_Sat(pert_deputy1[i,:3]-pert_chief[i,:3],pert_deputy1[i,3:]-pert_chief[i,3:],times[i],ref).to_Baseline(LVLH,Base))
    rel_p_dep2.append(orbits.LVLH_Sat(pert_deputy2[i,:3]-pert_chief[i,:3],pert_deputy2[i,3:]-pert_chief[i,3:],times[i],ref).to_Baseline(LVLH,Base))
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
    
    #Sum of the separation along the star direction and the baseline direction
    total_sep[ix] = baseline_sep[ix] + s_hat_sep[ix]

#Numerical differentiation twice - position -> acceleration
def acc(pos,times):
    vel = np.gradient(pos, times, edge_order=2)
    acc = np.gradient(vel, times, edge_order=2)
    return acc

#Accelerations - numerically integrate the position/time arrays found above
#Returns the absolute value of the acceleration in a given direction
acc_s1 = np.abs(acc(s_hat_drd1,times))
acc_s2 = np.abs(acc(s_hat_drd2,times))
acc_delta_b = np.abs(acc(baseline_sep,times))
acc_delta_s = np.abs(acc(s_hat_sep,times))
acc_total = np.abs(acc(total_sep,times))

#Maximum accelerations
max_acc_s1 = max(acc_s1)
max_acc_s2 = max(acc_s2)
max_acc_delta_b = max(acc_delta_b)
max_acc_delta_s = max(acc_delta_s)
max_acc_total = max(acc_total)

#Delta v (Integral of the absolute value of the acceleration)
delta_v_s1 = np.trapz(acc_s1)
delta_v_s2 = np.trapz(acc_s2)
delta_v_delta_b = np.trapz(acc_delta_b)
delta_v_delta_s = np.trapz(acc_delta_s)
delta_v_total = np.trapz(acc_total)

#Result array
#result[0] is the max a between deputy 1 and chief in the star direction
#result[1] is the max a between deputy 2 and chief in the star direction
#result[2] is the max a between the two deputies in the star direction (ie the difference between 0 and 1)
#result[3] is the max a in the baseline direction
#result[4] is the max total a (sum of 2 and 3; the total acceleration that needs to be corrected for)
#result[5-9] are the same, but for delta v

result = np.array([max_acc_s1,max_acc_s2,max_acc_delta_s,
                   max_acc_delta_b,max_acc_total,delta_v_s1,delta_v_s2,
                   delta_v_delta_s,delta_v_delta_b,delta_v_total])

# ---------------------------------------------------------------------- #
### PLOTTING STUFF ###

### Functions to set 3D axis aspect ratio as equal
def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)
"""
#Plot ECI Orbit
plt.figure(1)
plt.clf()
ax1 = plt.axes(projection='3d')
ax1.set_aspect('equal')
ax1.plot3D(ECI_rc[:,0],ECI_rc[:,1],ECI_rc[:,2],'b-')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('z (m)')
ax1.set_title('Orbit in ECI frame')
set_axes_equal(ax1)

#Plot perturbed LVLH orbits
plt.figure(2)
plt.clf()
ax2 = plt.axes(projection='3d')
ax2.set_aspect('equal')
ax2.plot3D(pert_LVLH_drd1[:,0],pert_LVLH_drd1[:,1],pert_LVLH_drd1[:,2],'b--')
ax2.plot3D(pert_LVLH_drd2[:,0],pert_LVLH_drd2[:,1],pert_LVLH_drd2[:,2],'c--')
ax2.set_xlabel('r (m)')
ax2.set_ylabel('v (m)')
ax2.set_zlabel('h (m)')
ax2.set_title('Orbit in LVLH frame')
set_axes_equal(ax2)
"""
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
