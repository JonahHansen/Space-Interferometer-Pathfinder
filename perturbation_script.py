from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
from orbits import ECI_orbit
from perturbations import dX_dt
from matplotlib.collections import LineCollection

plt.ion()

alt = 500e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(20) #20
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
ECI = ECI_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

#Number of orbits
n_orbits = 1
#Number of phases in each orbit
n_phases = 1000
#Total evaluation points
n_times = n_orbits*n_phases
times = np.linspace(0,ECI.period*n_orbits,n_times) #Create list of times

"""Initialise state arrays"""
ECI_rc = np.zeros((n_times,6)) #Chief ECI position vector
ECI_rd1 = np.zeros((n_times,6)) #Deputy 1 ECI position vector
ECI_rd2 = np.zeros((n_times,6)) #Deputy 2 ECI position vector
LVLH_drd1 = np.zeros((n_times,6)) #Deputy 1 LVLH position vector
LVLH_drd2 = np.zeros((n_times,6)) #Deputy 2 LVLH position vector
s_hats = np.zeros((n_times,3)) #Star vectors

#Calculate the positions of the chief and deputies in the absence of
#perturbations in both the ECI and LVLH frames
for i in range(n_times):
    ECI_rc[i] = ECI.chief_state(times[i])
    rot_mat = ECI.to_LVLH_mat(ECI_rc[i]) #Rotation matrix
    ECI_rd1[i] = ECI.deputy1_state(ECI_rc[i]) #Deputy 1 position
    ECI_rd2[i] = ECI.deputy2_state(ECI_rc[i]) #Deputy 2 position
    LVLH_drd1[i] = ECI.ECI_to_LVLH_state(ECI_rc[i],rot_mat,ECI_rd1[i])
    LVLH_drd2[i] = ECI.ECI_to_LVLH_state(ECI_rc[i],rot_mat,ECI_rd2[i])
    s_hats[i] = np.dot(rot_mat,ECI.s_hat) #Star vectors

#Tolerance and steps required for the integrator
rtol = 1e-9
atol = 1e-18
step = 10

#Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ECI,p_list), [times[0],times[-1]], LVLH_drd1[0], t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X_d1.success:
    raise Exception("Integration failed!!!!")

X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ECI,p_list), [times[0],times[-1]], LVLH_drd2[0], t_eval = times, rtol = rtol, atol = atol, max_step=step)
if not X_d2.success:
    raise Exception("Integration failed!!!!")

#Peturbed orbits
pert_LVLH_drd1 = np.transpose(X_d1.y)
pert_LVLH_drd2 = np.transpose(X_d2.y)

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
    baseline_sep[ix] = np.linalg.norm(pert_LVLH_drd1[ix,:3]) - np.linalg.norm(pert_LVLH_drd2[ix,:3])
    #Component of perturbed orbit in star direction
    s_hat_drd1[ix] = np.dot(pert_LVLH_drd1[ix,:3],s_hats[ix])
    s_hat_drd2[ix] = np.dot(pert_LVLH_drd2[ix,:3],s_hats[ix])
    #Baseline unit vector
    b_hat = pert_LVLH_drd1[ix,:3]/np.linalg.norm(pert_LVLH_drd1[ix,:3])
    #Component of perturbed orbit in baseline direction
    b_hat_drd1[ix] = np.dot(pert_LVLH_drd1[ix,:3],b_hat)
    b_hat_drd2[ix] = np.dot(pert_LVLH_drd2[ix,:3],b_hat)
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

#Plot ECI Orbit
plt.figure(1)
plt.clf()
ax1 = plt.axes(projection='3d')
ax1.set_aspect('equal')
ax1.plot3D(ECI_rc[:,0],ECI_rc[:,1],ECI_rc[:,2],'b-')
ax1.plot3D(ECI_rd1[:,0],ECI_rd1[:,1],ECI_rd1[:,2],'r-')
ax1.plot3D(ECI_rd2[:,0],ECI_rd2[:,1],ECI_rd2[:,2],'g-')
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

#Plot separation along the star direction
plt.figure(3)
plt.clf()
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
cbar.set_label('Time (s)', rotation=270, labelpad = 15)
