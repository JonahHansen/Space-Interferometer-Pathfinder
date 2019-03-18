from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from orbits import ECI_orbit
import quaternions as qt
from Schweighart_J2 import J2_pet

plt.ion()

alt = 500e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(90) #49
#Longitude of the Ascending Node
Om_0 = np.radians(50) #-30

#Stellar vector
ra = np.radians(30) #23
dec = np.radians(20)#43

#The max distance to the other satellites in m
delta_r_max = 0.1*1e3

#------------------------------------------------------------------------------------------
#Calculate orbit, in the geocentric (ECI) frame
ECI = ECI_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

num_times = 1000
times = np.linspace(0,ECI.period,num_times) #Create list of times

"""Initialise arrays"""
ECI_rc = np.zeros((num_times,6)) #Chief ECI position vector
ECI_rd1 = np.zeros((num_times,6)) #Deputy 1 ECI position vector
ECI_rd2 = np.zeros((num_times,6)) #Deputy 2 ECI position vector
LVLH_rc = np.zeros((num_times,6)) #Chief LVLH position vector
LVLH_rd1 = np.zeros((num_times,6)) #Deputy 1 LVLH position vector
LVLH_rd2 = np.zeros((num_times,6)) #Deputy 2 LVLH position vector
LVLH_drd1 = np.zeros((num_times,6)) #Deputy 1 LVLH separation vector
LVLH_drd2 = np.zeros((num_times,6)) #Deputy 2 LVLH separation vector
s_hats = np.zeros((num_times,3)) #Star vectors

i = 0
for t in times:
    ECI_rc[i] = ECI.chief_state(t)
    rot_mat = ECI.to_LVLH_mat(ECI_rc[i]) #Rotation matrix
    ECI_rd1[i] = ECI.deputy1_state(ECI_rc[i]) #Deputy 1 position
    ECI_rd2[i] = ECI.deputy2_state(ECI_rc[i]) #Deputy 2 position
    #LVLH vectors
    LVLH_rc[i] = ECI.to_LVLH_state(ECI_rc[i],rot_mat,ECI_rc[i])
    LVLH_rd1[i] = ECI.to_LVLH_state(ECI_rc[i],rot_mat,ECI_rd1[i])
    LVLH_rd2[i] = ECI.to_LVLH_state(ECI_rc[i],rot_mat,ECI_rd2[i])
    #Separation vectors
    LVLH_drd1[i] = LVLH_rd1[i] - LVLH_rc[i]
    LVLH_drd2[i] = LVLH_rd2[i] - LVLH_rc[i]
    s_hats[i] = np.dot(rot_mat,ECI.s_hat) #Star vectors
    i += 1

rtol = 1e-6
atol = 1e-12
step = 100

"""
Use Schweighart J2 formula
func1 = J2_pet(LVLH_sep_state1[0],ECI,ECI.q1)
func2 = J2_pet(LVLH_sep_state2[0],ECI,ECI.q2)

X_d1 = solve_ivp(func1, [times[0],times[-1]], LVLH_drd1[0], t_eval = times, rtol = rtol, atol = atol, max_step=step)
X_d2 = solve_ivp(func2, [times[0],times[-1]], LVLH_drd2[0], t_eval = times, rtol = rtol, atol = atol, max_step=step)
"""

""" Differential equation function"""
def dX_dt(t, state, ECI):
    r = state[:3] #Position
    v = state[3:] #Velocity

    #First half of the differential vector
    dX0 = v[0]
    dX1 = v[1]
    dX2 = v[2]

    #Position in LVLH frame, with origin at centre of the Earth
    rd = np.array([ECI.R_orb+r[0],r[1],r[2]])

    n = ECI.ang_vel #Angular velocity
    mu = const.GM_earth.value #Graviational parameter
    omega = np.array([0,0,n]) #Angular velocity vector in LVLH frame

    """ J2 Acceleration"""
    c_state = ECI.chief_state(t) #Chief state in ECI at time t
    mat = ECI.to_LVLH_mat(c_state) #Matrix to convert into LVLH

    J2 = 0.00108263 #J2 Parameter
    [x,y,z] = np.dot(np.linalg.inv(mat),r[:3]) + c_state[:3] #Deputy position in ECI coordinates

    #Calculate J2 acceleration from the equation in ECI frame
    J2_fac1 = 3/2*J2*mu*const.R_earth.value**2/ECI.R_orb**5
    J2_fac2 = 5*z**2/ECI.R_orb**2
    J2_pet = J2_fac1*np.array([x*(J2_fac2-1),y*(J2_fac2-1),z*(J2_fac2-3)])

    #Convert back to LVLH frame
    J2_pet_LVLH = np.dot(mat,J2_pet)
    J2_pet_LVLH = 0 #Comment out to use J2

    #HCW Equations
    K = np.diag(np.array([3*n**2,0,-(n**2)]))
    Gamma2 = n**2/ECI.R_orb*np.array([-3*r[0]**2 + 1.5*r[1]**2 + 1.5*r[2]**2, 3*r[0]*r[1], 3*r[0]*r[2]])
    a = -2*np.cross(omega,v) + np.matmul(K,r) + Gamma2 + J2_pet_LVLH

    #Acceleration vector - analytical version (See Butcher 18)
    #a = -2*np.cross(omega,v) - np.cross(omega,np.cross(omega,rd)) - mu*rd/np.linalg.norm(rd)**3 + J2_pet_LVLH

    #Second half of the differential vector
    dX3 = a[0]
    dX4 = a[1]
    dX5 = a[2]
    return np.array([dX0,dX1,dX2,dX3,dX4,dX5])

#Integrate the orbits
X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ECI), [times[0],times[-1]], LVLH_drd1[0], t_eval = times, rtol = rtol, atol = atol, max_step=step)
X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ECI), [times[0],times[-1]], LVLH_drd2[0], t_eval = times, rtol = rtol, atol = atol, max_step=step)

#Peturbed orbits
pert_LVLH_drd1 = np.transpose(X_d1.y)
pert_LVLH_drd2 = np.transpose(X_d2.y)

baseline_sep = np.zeros(num_times) #Separation along the baseline
s_hat_drd1 = np.zeros(num_times) #Deputy1 position in star direction
s_hat_drd2 = np.zeros(num_times) #Deputy2 position in star direction
s_hat_sep = np.zeros(num_times) #Separation along the baseline
total_sep = np.zeros(num_times) #Total separation

for ix in range(len(pert_LVLH_drd1)):
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
acc_s1 = acc(s_hat_drd1)
acc_s2 = acc(s_hat_drd2)
acc_delta_b = acc(baseline_sep)
acc_delta_s = acc(s_hat_sep)
acc_total = acc(total_sep)

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
plt.plot(times,baseline_sep,"y-",label="Separation, baseline direction")
plt.plot(times,total_sep,"c-",label="Total direction")
plt.xlabel("Times(s)")
plt.ylabel("Separation(m)")
plt.title('Separations against time due to perturbations')
plt.legend()
