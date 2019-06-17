from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits2 as orb
from matplotlib.collections import LineCollection
from modules.Schweighart_J2 import J2_pet
#from scipy.optimize import minimize

plt.ion()

alt = 500e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(39) #20
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
ECI = orb.ECI_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

#Number of orbits
n_orbits = 0.5
#Number of phases in each orbit
n_phases = 20000
#Total evaluation points
n_times = int(n_orbits*n_phases)
times = np.linspace(0,ECI.period*n_orbits,n_times) #Create list of times

"""Initialise state arrays"""
ECI_rc = np.zeros((n_times,6)) #Chief state
s_hats = np.zeros((n_times,3)) #Star vectors

#Calculate the positions of the chief and deputies in the absence of
#perturbations in both the ECI and LVLH frames
for i in range(n_times):
    chief = orb.init_chief(ECI,times[i],True)
    ECI_rc[i] = chief.state
    s_hats[i] = np.dot(chief.LVLHmat,ECI.s_hat) #Star vectors
chief_0 = orb.init_chief(ECI,0)
LVLH_drd1_0 = orb.init_deputy(ECI,chief_0,1).to_LVLH(chief_0)
LVLH_drd2_0 = orb.init_deputy(ECI,chief_0,2).to_LVLH(chief_0)

J2_func1 = J2_pet(LVLH_drd1_0,ECI)
J2_func2 = J2_pet(LVLH_drd2_0,ECI)

#Tolerance and steps required for the integrator
rtol = 1e-9
atol = 1e-18
step = 10

def ECI_chief_pert(t,state):
    [x,y,z,dx,dy,dz] = state

    J2 = 0.00108263 #J2 Parameter

    J2_fac1 = 3/2*J2*const.GM_earth.value*const.R_earth.value**2/R_orb**5

    #Calculate J2 acceleration for chief satellite
    J2_fac2 = 5*z**2/R_orb**2
    J2_p = J2_fac1*np.array([x*(J2_fac2-1),y*(J2_fac2-1),z*(J2_fac2-3)])

    g = -const.GM_earth.value/R_orb**3*(np.array([x,y,z]))

    [ddx,ddy,ddz] = g + J2_p

    return np.array([dx,dy,dz,ddx,ddy,ddz])

X_c = solve_ivp(ECI_chief_pert, [times[0],times[-1]], chief_0.state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
if not X_c.success:
    raise Exception("Integration failed!!!!")
pert_chief = np.transpose(X_c.y)

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

t_burn = 0.5 #How long between corrections in seconds


times_lsls = list(chunktime(times,t_burn)) #List of times

state1 = LVLH_drd1_0.state #Initial state
state2 = LVLH_drd2_0.state #Initial state

last_time = 0

#Function to determine delta v.
def integrate_delv_burn(t,pt,state1,state2,prev1,prev2,delv_ls):

    index = np.where(times==t)[0][0]
    pindex = np.where(times==pt)[0][0]

    c = orb.Chief(ECI,pert_chief[index,:3],pert_chief[index,3:],ECI.q0)
    c_s_hat = np.dot(c.LVLHmat,ECI.s_hat)
    b1 = orb.LVLH_Deputy(state1[:3],state1[3:],LVLH_drd1_0.q,c_s_hat).to_Baseline(state2[:3]-state1[:3])
    b2 = orb.LVLH_Deputy(state2[:3],state2[3:],LVLH_drd2_0.q,c_s_hat).to_Baseline(state2[:3]-state1[:3])

    pc = orb.Chief(ECI,pert_chief[pindex,:3],pert_chief[pindex,3:],ECI.q0)
    pc_s_hat = np.dot(pc.LVLHmat,ECI.s_hat)
    pb1 = orb.LVLH_Deputy(prev1[:3],prev1[3:],LVLH_drd1_0.q,pc_s_hat).to_Baseline(prev2[:3]-prev1[:3])
    pb2 = orb.LVLH_Deputy(prev2[:3],prev2[3:],LVLH_drd2_0.q,pc_s_hat).to_Baseline(prev2[:3]-prev1[:3])

    delv1 = np.zeros(3)
    delv2 = np.zeros(3)

    delv1[2] = -0.5*(b1.pos[2] - pb1.pos[2])/(t-pt)
    delv2[2] = -0.5*(b2.pos[2] - pb2.pos[2])/(t-pt)

    delta_b = -((b1.pos[0] - pb1.pos[0]) + (b2.pos[0] - pb2.pos[0]))/2
    delv1[0] = delta_b
    delv2[0] = delta_b

    delv1_LVLH = orb.Baseline_Deputy(np.zeros(3),delv1,LVLH_drd1_0.q,b1.basemat).to_LVLH(c)
    delv2_LVLH = orb.Baseline_Deputy(np.zeros(3),delv2,LVLH_drd2_0.q,b2.basemat).to_LVLH(c)

    print(delv1_LVLH.vel)

    #print(delv1_LVLH.vel,delv2_LVLH.vel)

    state1[3:] += delv1_LVLH.vel
    state2[3:] += delv2_LVLH.vel

    #Delta vs for each satellite
    delv_ls.append(list(delv1)+list(delv2))

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

    #Perform the burn
    state1,state2,delv_ls = integrate_delv_burn(time[-1],last_time,pert_LVLH_drd1[-1],pert_LVLH_drd2[-1],state1,state2,delv_ls)

    last_time = time[-1] #Last time of this round, to use in the next round

total_sep = np.zeros(n_times) #Total separation

for ix in range(n_times):
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
