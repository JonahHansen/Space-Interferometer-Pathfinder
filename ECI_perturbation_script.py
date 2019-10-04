from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits as orbits
from matplotlib.collections import LineCollection
from modules.Schweighart_J2_solved import propagate_spacecraft
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
n_orbits = 0.5
#Number of phases in each orbit
n_phases = 100
#Total evaluation points
n_times = int(n_orbits*n_phases)
times = np.linspace(0,ref.periodK*n_orbits,n_times) #Create list of times

#Initial reference orbit state
pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(0)

precession = False

#Initial states of the satellites
chief_0 = orbits.init_chief(ref,precession)
deputy1_0 = orbits.init_deputy(ref,1,precession)
deputy2_0 = orbits.init_deputy(ref,2,precession)

### ECI version ###

#Tolerance and steps required for the integrator
rtol = 1e-12
atol = 1e-18

def dX_dt(t,state,ref):
    [x,y,z] = state[:3] #Position
    v = state[3:] #Velocity

    #First half of the differential vector (derivative of position, velocity)
    dX0 = v[0]
    dX1 = v[1]
    dX2 = v[2]

    J2 = 0.00108263 #J2 Parameter

    r = np.linalg.norm(state[:3])

    #Calculate J2 acceleration from the equation in ECI frame
    J2_fac1 = 3/2*J2*const.GM_earth.value*const.R_earth.value**2/r**5
    J2_fac2 = 5*z**2/r**2
    J2_p = J2_fac1*np.array([x*(J2_fac2-1),y*(J2_fac2-1),z*(J2_fac2-3)])

    r_hat = state[:3]/r
    a = -const.GM_earth.value/r**2*r_hat + J2_p
    dX3 = a[0]
    dX4 = a[1]
    dX5 = a[2]
    
    print(1/2*(np.linalg.norm(v)**2) - const.GM_earth.value/r)

    return np.array([dX0,dX1,dX2,dX3,dX4,dX5])

#Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
X_c = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], chief_0.state, t_eval = times, rtol = rtol, atol = atol)
#Check if successful integration
if not X_c.success:
    raise Exception("Integration Chief failed!!!!")

#Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], deputy1_0.state, t_eval = times, rtol = rtol, atol = atol)
#Check if successful integration
if not X_d1.success:
    raise Exception("Integration Deputy 1 failed!!!!")

X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], deputy2_0.state, t_eval = times, rtol = rtol, atol = atol)
if not X_d2.success:
    raise Exception("Integration Deputy 2 failed!!!!")

chief_p_states = X_c.y.transpose()
deputy1_p_states = X_d1.y.transpose()
deputy2_p_states = X_d2.y.transpose()

c_sats = []
d1_sats = []
d2_sats = []

print("Integration Done")
for i in range(len(times)):
    c_sats.append(orbits.ECI_Sat(chief_p_states[i,:3],chief_p_states[i,3:],times[i],ref).to_Baseline(state=chief_p_states[i]))
    d1_sats.append(orbits.ECI_Sat(deputy1_p_states[i,:3],deputy1_p_states[i,3:],times[i],ref).to_Baseline(state=chief_p_states[i]))
    d2_sats.append(orbits.ECI_Sat(deputy2_p_states[i,:3],deputy2_p_states[i,3:],times[i],ref).to_Baseline(state=chief_p_states[i]))
print("Classifying Done")

d1_rel_pos = np.zeros((n_times,3)) #Deputy1 position in star direction
d2_rel_pos = np.zeros((n_times,3)) #Deputy2 position in star direction

for ix in range(n_times):
    d1_rel_pos[ix] = d1_sats[ix].pos - c_sats[ix].pos
    d2_rel_pos[ix] = d2_sats[ix].pos - c_sats[ix].pos

plt.figure(1)
plt.clf()

plt.subplot(3,1,1)
plt.plot(times,d1_rel_pos[:,0],'b-',label="Deputy1")
plt.plot(times,d2_rel_pos[:,0],'r-',label="Deputy2")
plt.ylabel(r"$\rho$ Separation (m)")
plt.title("Separations in the LVLH Frame \n "+r"$i = %d\degree$, $\Omega = %d\degree$, $\alpha = %d\degree$, $\delta = %d\degree$"%(np.degrees(inc_0),np.degrees(Om_0),np.degrees(ra),np.degrees(dec)))

plt.legend()
plt.subplot(3,1,2)
plt.plot(times,d1_rel_pos[:,1],'b-',label="Deputy1")
plt.plot(times,d2_rel_pos[:,1],'r-',label="Deputy2")
plt.ylabel(r"$\xi$ Separation (m)")


plt.subplot(3,1,3)
plt.plot(times,d1_rel_pos[:,2],'b-',label="Deputy1")
plt.plot(times,d2_rel_pos[:,2],'r-',label="Deputy2")
plt.ylabel(r"$\eta$ Separation (m)")
plt.xlabel("Time (s)")

#plt.savefig('ECI_Curvy_45_long.svg', format='svg')







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
    baseline_sep[ix] = np.linalg.norm(d2_rel_pos[ix]) - np.linalg.norm(d1_rel_pos[ix])

    #Component of perturbed orbit in star direction
    s_hat_drd1[ix] = d1_rel_pos[ix,2]
    s_hat_drd2[ix] = d2_rel_pos[ix,2]

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

#Maximum accelerations
max_acc_s1 = max(acc_s1)
max_acc_s2 = max(acc_s2)
max_acc_delta_b = max(acc_delta_b)

#Delta v (Integral of the absolute value of the acceleration)
delta_v_s1 = np.trapz(acc_s1,times)
delta_v_s2 = np.trapz(acc_s2,times)
delta_v_delta_b = np.trapz(acc_delta_b,times)

print("Delv delb: " + str(delta_v_delta_b))
print("Delv s1: " + str(delta_v_s1))

max_acc = np.array([0.,max_acc_delta_b + max_acc_s2,max_acc_delta_b + max_acc_s2]) + max_acc_s1
delta_v = np.array([0.,delta_v_delta_b + delta_v_s2,delta_v_delta_b + delta_v_s2]) + delta_v_s1
percent = delta_v/np.array([0.04,0.08,0.08])*100

#Result array
#result[0] is the max a between deputy 1 and chief in the star direction
#result[1] is the max a between deputy 2 and chief in the star direction
#result[2] is the max a between the two deputies in the star direction (ie the difference between 0 and 1)
#result[3] is the max a in the baseline direction
#result[4] is the max total a (sum of 2 and 3; the total acceleration that needs to be corrected for)
#result[5-9] are the same, but for delta v

result = np.array([max_acc,delta_v,percent])
print(result)
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
"""
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
"""
