from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
from orbits import ECI_orbit
from perturbations import dX_dt
#import quaternions as qt

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
delta_r_max = 0.3*1e3

#List of perturbations: 1 = J2, 2 = Solar radiation, 3 = Drag. Leave empty list if no perturbations.
p_list = [1]

#------------------------------------------------------------------------------------------
#Calculate orbit, in the geocentric (ECI) frame
ECI = ECI_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

num_times = 1000
n_periods = 1
times = np.linspace(0,ECI.period*n_periods,num_times) #Create list of times

"""Initialise arrays"""
ECI_rc = np.zeros((num_times,6)) #Chief ECI position vector
ECI_rd1 = np.zeros((num_times,6)) #Deputy 1 ECI position vector
ECI_rd2 = np.zeros((num_times,6)) #Deputy 2 ECI position vector
s_hats = np.zeros((num_times,3)) #Star vectors

i = 0
for t in times:
    ECI_rc[i] = ECI.chief_state(t)
    rot_mat = ECI.to_LVLH_mat(ECI_rc[i]) #Rotation matrix
    ECI_rd1[i] = ECI.deputy1_state(ECI_rc[i]) #Deputy 1 position
    ECI_rd2[i] = ECI.deputy2_state(ECI_rc[i]) #Deputy 2 position
    s_hats[i] = np.dot(rot_mat,ECI.s_hat) #Star vectors
    i += 1

rot_mat = ECI.to_LVLH_mat(ECI_rc[0])
LVLH_drd1_0 = ECI.ECI_to_LVLH_state(ECI_rc[0],rot_mat,ECI_rd1[0]) #Initial LVLH separation state for deputy 1
LVLH_drd2_0 = ECI.ECI_to_LVLH_state(ECI_rc[0],rot_mat,ECI_rd2[0]) #Initial LVLH separation state for deputy 1

rtol = 1e-9
atol = 1e-18
step = 100

#Integrate the orbits
X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ECI,p_list), [times[0],times[-1]], LVLH_drd1_0, t_eval = times, rtol = rtol, atol = atol, max_step=step)
X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ECI,p_list), [times[0],times[-1]], LVLH_drd2_0, t_eval = times, rtol = rtol, atol = atol, max_step=step)

#Peturbed orbits
pert_LVLH_drd1 = np.transpose(X_d1.y)
pert_LVLH_drd2 = np.transpose(X_d2.y)

#--------------------------------------------------------------------------------------------- #

baseline_sep = np.zeros(num_times) #Separation along the baseline
s_hat_drd1 = np.zeros(num_times) #Deputy1 position in star direction
s_hat_drd2 = np.zeros(num_times) #Deputy2 position in star direction
s_hat_sep = np.zeros(num_times) #Separation along the baseline
total_sep = np.zeros(num_times) #Total separation

for ix in range(num_times):
    baseline_sep[ix] = np.linalg.norm(pert_LVLH_drd1[ix]) - np.linalg.norm(pert_LVLH_drd2[ix])
    s_hat_drd1[ix] = np.dot(pert_LVLH_drd1[ix,:3],s_hats[ix])
    s_hat_drd2[ix] = np.dot(pert_LVLH_drd2[ix,:3],s_hats[ix])
    s_hat_sep[ix] = s_hat_drd1[ix] - s_hat_drd2[ix]
    total_sep[ix] = baseline_sep[ix] + s_hat_sep[ix]

#Numerical differentiation
def acc(pos):
    vel = np.gradient(pos, edge_order=2)
    acc = np.gradient(vel, edge_order=2)
    return np.abs(acc)

#Accelerations
acc_s1 = np.abs(acc(s_hat_drd1))
acc_s2 = np.abs(acc(s_hat_drd2))
acc_delta_b = np.abs(acc(baseline_sep))
acc_delta_s = np.abs(acc(s_hat_sep))
acc_total = np.abs(acc(total_sep))

#Maximum accelerations
max_acc_s1 = max(acc_s1)
max_acc_s2 = max(acc_s2)
max_acc_delta_b = max(acc_delta_b)
max_acc_delta_s = max(acc_delta_s)
max_acc_total = max(acc_total)

#Delta v
delta_v_s1 = np.trapz(acc_s1)
delta_v_s2 = np.trapz(acc_s2)
delta_v_delta_b = np.trapz(acc_delta_b)
delta_v_delta_s = np.trapz(acc_delta_s)
delta_v_total = np.trapz(acc_total)

#Result array
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

plt.figure(3)
plt.clf()
plt.plot(times,s_hat_drd1,"b-",label="Deputy 1, s direction")
plt.plot(times,s_hat_drd2,"g-",label="Deputy 2, s direction")
plt.plot(times,s_hat_sep,"r-",label="Separation, s direction")
#plt.plot(times,baseline_sep,"y-",label="Separation, baseline direction")
#plt.plot(times,total_sep,"c-",label="Total direction")
plt.xlabel("Times(s)")
plt.ylabel("Separation(m)")
plt.title('Separations against time due to perturbations')
plt.legend()

plt.figure(4)
plt.clf()
#plt.plot(times,s_hat_drd1,"b-",label="Deputy 1, s direction")
#plt.plot(times,s_hat_drd2,"g-",label="Deputy 2, s direction")
#plt.plot(times,s_hat_sep,"r-",label="Separation, s direction")
plt.plot(times,baseline_sep,"y-",label="Separation, baseline direction")
#plt.plot(times,total_sep,"c-",label="Total direction")
plt.xlabel("Times(s)")
plt.ylabel("Separation(m)")
plt.title('Separations against time due to perturbations')
plt.legend()
