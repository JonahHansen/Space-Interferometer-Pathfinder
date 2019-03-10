from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
from orbits2 import ECI_orbit

plt.ion()

""" Differential equation function with NO J2 perturbation """
def dX_dt(t, state, ECI):
    r = state[:3] #Position
    v = state[3:] #Velocity

    dX0 = v[0]
    dX1 = v[1]
    dX2 = v[2]

    mat = ECI.to_LVLH_mat(ECI.chief_state(t))

    n = ECI.ang_vel #Angular velocity
    mu = const.GM_earth.value #Graviational parameter
    omega = np.array([0,0,n]) #Angular velocity vector in LVLH frame

    print(t)

    #HCW Equations - Until this works, will use the analytical form
    K = np.diag(np.array([3*n**2,0,-(n**2)]))
    Gamma2 = n**2/ECI.R_orb*np.array([-3*r[0]**2 + 1.5*r[1]**2 + 1.5*r[2]**2, 3*r[0]*r[1], 3*r[0]*r[2]])
    a = -2*np.cross(omega,v) + np.matmul(K,r) + Gamma2

    #Position vector of deputy
    #rd = np.array([LVLH_orbit.R_orb+r[0],r[1],r[2]])
    #Acceleration vector - analytical version (See Butcher 18)
    # a = -2*np.cross(omega,v) - np.cross(omega,np.cross(omega,rd)) - mu*np.array([-2*r[0],r[1],r[2]])/np.linalg.norm(rd)**3

    dX3 = a[0]
    dX4 = a[1]
    dX5 = a[2]
    return np.array([dX0,dX1,dX2,dX3,dX4,dX5])


alt = 1000e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(2) #49
#Longitude of the Ascending Node
Om_0 = np.radians(0) #-30

#Stellar vector
ra = np.radians(0) #23
dec = np.radians(90)#43

#The max distance to the other satellites in m
delta_r_max = 0.3*1e3

#------------------------------------------------------------------------------------------
#Calculate orbit, in the geocentric (ECI) frame
ECI = ECI_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

num_times = 1000
times = np.linspace(0,ECI.period,num_times)

c_state = np.zeros((num_times,6))
dep1_state = np.zeros((num_times,6))
dep2_state = np.zeros((num_times,6))
LVLH_state0 = np.zeros((num_times,6))
LVLH_state1 = np.zeros((num_times,6))
LVLH_state2 = np.zeros((num_times,6))
LVLH_sep_state1 = np.zeros((num_times,6))
LVLH_sep_state2 = np.zeros((num_times,6))

i = 0
for t in times:
    c_state[i] = ECI.chief_state(t)
    dep1_state[i] = ECI.deputy1_state(c_state[i])
    dep2_state[i] = ECI.deputy2_state(c_state[i])
    LVLH_state0[i] = ECI.to_LVLH_state(c_state[i],c_state[i])
    LVLH_state1[i] = ECI.to_LVLH_state(c_state[i],dep1_state[i])
    LVLH_state2[i] = ECI.to_LVLH_state(c_state[i],dep2_state[i])
    LVLH_sep_state1[i] = LVLH_state1[i] - LVLH_state0[i]
    LVLH_sep_state2[i] = LVLH_state2[i] - LVLH_state0[i]
    i += 1

rtol = 1e-6
atol = 1e-12
step = 100

#Integrate the orbits
X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ECI), [times[0],times[-1]], LVLH_sep_state1[0], t_eval = times, rtol = rtol, atol = atol, max_step=step)
X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ECI), [times[0],times[-1]], LVLH_sep_state2[0], t_eval = times, rtol = rtol, atol = atol, max_step=step)

pert_LVLH_sep_state1 = np.transpose(X_d1.y)
pert_LVLH_sep_state2 = np.transpose(X_d2.y)

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
ax1 = plt.axes(projection='3d')
ax1.set_aspect('equal')
ax1.plot3D(c_state[:,0],c_state[:,1],c_state[:,2],'b-')
ax1.plot3D(dep1_state[:,0],dep1_state[:,1],dep1_state[:,2],'r-')
ax1.plot3D(dep2_state[:,0],dep2_state[:,1],dep2_state[:,2],'g-')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('z (m)')
ax1.set_title('Orbit in ECI frame')
set_axes_equal(ax1)

#Plot LVLH (solid) and perturbed LVLH (dashed) orbits
plt.figure(2)
ax2 = plt.axes(projection='3d')
ax2.set_aspect('equal')
ax2.plot3D(LVLH_state0[:,0],LVLH_state0[:,1],LVLH_state0[:,2],'b-')
ax2.plot3D(LVLH_state1[:,0],LVLH_state1[:,1],LVLH_state1[:,2],'r-')
ax2.plot3D(LVLH_state2[:,0],LVLH_state2[:,1],LVLH_state2[:,2],'g-')
ax2.plot3D(pert_LVLH_sep_state1[:,0],pert_LVLH_sep_state1[:,1],pert_LVLH_sep_state1[:,2],'r--')
ax2.plot3D(pert_LVLH_sep_state2[:,0],pert_LVLH_sep_state2[:,1],pert_LVLH_sep_state2[:,2],'g--')
ax2.set_xlabel('r (m)')
ax2.set_ylabel('v (m)')
ax2.set_zlabel('h (m)')
ax2.set_title('Orbit in LVLH frame')
set_axes_equal(ax2)
