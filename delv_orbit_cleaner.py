from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
from modules.orbits import ECI_orbit, Chief, init_deputy
from matplotlib.collections import LineCollection
from modules.Schweighart_J2 import J2_pet
#from scipy.optimize import minimize

plt.ion()

alt = 500e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(97) #20
#Longitude of the Ascending Node
Om_0 = np.radians(3) #0

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
n_orbits = 0.5
#Number of phases in each orbit
n_phases = 10000
#Total evaluation points
n_times = int(n_orbits*n_phases)
times = np.linspace(0,ECI.period*n_orbits,n_times) #Create list of times

"""Initialise state arrays"""
ECI_rc = np.zeros((n_times,6)) #Chief state
s_hats = np.zeros((n_times,3)) #Star vectors

#Calculate the positions of the chief and deputies in the absence of
#perturbations in both the ECI and LVLH frames
for i in range(n_times):
    chief = Chief(ECI,times[i],True)
    ECI_rc[i] = chief.state
    s_hats[i] = np.dot(chief.mat,ECI.s_hat) #Star vectors
chief_0 = Chief(ECI,0)
LVLH_drd1_0 = init_deputy(ECI,chief_0,1).to_LVLH(chief_0)
LVLH_drd2_0 = init_deputy(ECI,chief_0,2).to_LVLH(chief_0)

J2_func1 = J2_pet(LVLH_drd1_0,ECI)
J2_func2 = J2_pet(LVLH_drd2_0,ECI)

#Tolerance and steps required for the integrator
rtol = 1e-9
atol = 1e-18
step = 10

#Take a list of times and split it into chunks of "t"s
def chunktime(l, t):
    # For item i in a range that is a length of l,
    n = round(t*n_times/(ECI.period*n_orbits))
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

delv_ls = [] #List of delta vs
pert_LVLH_drd1 = np.zeros((0,6)) #Empty perturbed arrays
pert_LVLH_drd2 = np.zeros((0,6))

t_burn = 60 #How long between corrections in seconds

#Burn coefficients
kappa1 = 0.5
kappa2 = 0.5
kappa3 = 0

params = [kappa1,kappa2,kappa3,t_burn]

times_lsls = list(chunktime(times,t_burn)) #List of times

state1 = LVLH_drd1_0.state #Initial state
state2 = LVLH_drd2_0.state #Initial state

last_time = 0

#Function to determine delta v.
def integrate_delv_burn(params,t,state1,state2,delv_ls):
    kappa1,kappa2,kappa3,t_burn = params

    s_hat = np.dot(Chief(ECI,t,True).mat,ECI.s_hat) #Star vector at time t

    #Position and velocities of deputies
    pos1 = state1[:3]
    pos2 = state2[:3]
    vel1 = state1[3:]
    vel2 = state2[3:]

    pert1 = []
    J2_func1(t,state1,pert1)
    pert2 = []
    J2_func1(t,state1,pert2)

    pert_s1 = -kappa3*np.dot(pert1,s_hat)*t_burn
    pert_s2 = -kappa3*np.dot(pert2,s_hat)*t_burn

    #Component of position in star direction
    del_s1 = np.dot(pos1,s_hat)*s_hat
    del_s2 = np.dot(pos2,s_hat)*s_hat

    #Component of velocity in star direction
    #del_sv1 = np.dot(vel1,s_hat)*s_hat
    #del_sv2 = np.dot(vel2,s_hat)*s_hat

    #Calculate burn to get to a star position of 0 at the next time step
    delv_s1 = kappa1/t_burn*(0-del_s1)
    delv_s2 = kappa1/t_burn*(0-del_s2)

    #Remove star position
    new_pos1 = pos1 - del_s1
    new_pos2 = pos2 - del_s2

    #Remove star velocity
    #new_vel1 = vel1 - del_sv1
    #new_vel2 = vel2 - del_sv2

    #Calculate position vector to the midpoint of the two deputies
    del_b = new_pos2 - new_pos1 #Separation vector
    del_b_half = 0.5*del_b #Midpoint
    m0 = new_pos1 + del_b_half #Midpoint from centre

    #Calculate velocity vector to the midpoint of the two deputies
    #del_bv = new_vel1 - new_vel2
    #del_bv_half = 0.5*del_bv
    #mv0 = (new_vel1 + del_bv_half)

    #Calculate chief burn to the midpoint of the baseline
    delv_bc = kappa2/t_burn*(m0)
    #Negative burn for the deputies
    delv_bd = kappa2/t_burn*(-m0)

    #delv_bvc = kappa3*mv0
    #delv_bvd = kappa3*-mv0

    vel1 += delv_bd + pert_s1 + delv_s1# + delv_bvd #New velocity
    vel2 += delv_bd + pert_s2 + delv_s2# + delv_bvd #New velocity

    #New states
    state1 = np.concatenate((pos1,vel1))
    state2 = np.concatenate((pos2,vel2))

    #Delta vs for each satellite
    delv_ls.append([delv_bc,delv_s1,delv_s2])

    return state1,state2,delv_ls

for time in times_lsls:
    print(last_time)
    #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
    X_d1 = solve_ivp(J2_func1, [last_time,time[-1]], state1, t_eval = time, rtol = rtol, atol = atol, max_step=step)
    #Check if successful integration
    if not X_d1.success:
        raise Exception("Integration failed!!!!")

    X_d2 = solve_ivp(J2_func2, [last_time,time[-1]], state2, t_eval = time, rtol = rtol, atol = atol, max_step=step)
    if not X_d2.success:
        raise Exception("Integration failed!!!!")

    #Add states to the array
    pert_LVLH_drd1=np.concatenate((pert_LVLH_drd1,np.transpose(X_d1.y)))
    pert_LVLH_drd2=np.concatenate((pert_LVLH_drd2,np.transpose(X_d2.y)))

    last_time = time[-1] #Last time of this round, to use in the next round

    #Perform the burn
    state1,state2,delv_ls = integrate_delv_burn(params,last_time,pert_LVLH_drd1[-1],pert_LVLH_drd2[-1],delv_ls)

total_sep = np.zeros(n_times) #Total separation

for ix in range(n_times//2):
    #Baseline separations is simply the difference between the positions of the two deputies
    baseline_sep = np.linalg.norm(pert_LVLH_drd1[ix,:3]) - np.linalg.norm(pert_LVLH_drd2[ix,:3])
    #Component of perturbed orbit in star direction
    s_hat_drd1 = np.dot(pert_LVLH_drd1[ix,:3],s_hats[ix])
    s_hat_drd2 = np.dot(pert_LVLH_drd2[ix,:3],s_hats[ix])
    #Separation of the two deputies in the star direction
    s_hat_sep = s_hat_drd1 - s_hat_drd2
    #Sum of the separation along the star direction and the baseline direction
    total_sep[ix] = baseline_sep + s_hat_sep

max_sep = np.max(np.abs(total_sep)) #Maximum total separation

#Norm the delta v
normed_delvs = [[np.linalg.norm(x) for x in delv_ls[j]] for j in range(len(delv_ls))]

delv_sums = np.sum(normed_delvs,axis=0) #Sum of the delta v for each satellite
delv_sum = np.sum(delv_sums) #Total delta v sum

print("max_sep = %.5f, delv = %s"%(max_sep,delv_sums))

cost_func = 0.1*delv_sum + 0.2*max_sep # Cost function

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
