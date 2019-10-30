""" Script to plot the J2 perturbed motion, calculated numerically in the ECI frame """
""" Takes in inc, Om, ra, dec from the command line """

import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits as orbits
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
n_orbits = 1
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


"""
J2 equation derivative equation for integrator
Inputs:
    t - time
    state - state of satellite
    ref - reference orbit
Outputs:
    derivative of state vector
"""
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

    return np.array([dX0,dX1,dX2,dX3,dX4,dX5])

#Integrate the orbits
X_c = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], chief_0.state, t_eval = times, rtol = rtol, atol = atol)
#Check if successful integration
if not X_c.success:
    raise Exception("Integration Chief failed!!!!")

#Integrate the orbits
X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], deputy1_0.state, t_eval = times, rtol = rtol, atol = atol)
#Check if successful integration
if not X_d1.success:
    raise Exception("Integration Deputy 1 failed!!!!")

X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], deputy2_0.state, t_eval = times, rtol = rtol, atol = atol)
if not X_d2.success:
    raise Exception("Integration Deputy 2 failed!!!!")

#List of perturbed states
chief_p_states = X_c.y.transpose()
deputy1_p_states = X_d1.y.transpose()
deputy2_p_states = X_d2.y.transpose()

c_sats = []
d1_sats = []
d2_sats = []

print("Integration Done")

#Convert into satellite class and change frame
for i in range(len(times)):
    c_sats.append(orbits.ECI_Sat(chief_p_states[i,:3],chief_p_states[i,3:],times[i],ref).to_Baseline(state=chief_p_states[i]))
    d1_sats.append(orbits.ECI_Sat(deputy1_p_states[i,:3],deputy1_p_states[i,3:],times[i],ref).to_Baseline(state=chief_p_states[i]))
    d2_sats.append(orbits.ECI_Sat(deputy2_p_states[i,:3],deputy2_p_states[i,3:],times[i],ref).to_Baseline(state=chief_p_states[i]))

#Positions relative to the chief
d1_rel_pos = np.zeros((n_times,3))
d2_rel_pos = np.zeros((n_times,3))
for ix in range(n_times):
    d1_rel_pos[ix] = d1_sats[ix].pos - c_sats[ix].pos
    d2_rel_pos[ix] = d2_sats[ix].pos - c_sats[ix].pos

#Plot the motion
plt.figure(1)
plt.clf()
plt.tight_layout()

plt.subplot(3,1,1)
plt.plot(times,d1_rel_pos[:,0],'b-',label="Deputy1")
plt.plot(times,d2_rel_pos[:,0],'r-',label="Deputy2")
plt.ylabel(r"$b$ Separation (m)")
plt.title("Separations in the Baseline Frame \n "+r"$i = %d\degree$, $\Omega = %d\degree$, $\alpha = %d\degree$, $\delta = %d\degree$"%(np.degrees(inc_0),np.degrees(Om_0),np.degrees(ra),np.degrees(dec)))

plt.legend()
plt.subplot(3,1,2)
plt.plot(times,d1_rel_pos[:,1],'b-',label="Deputy1")
plt.plot(times,d2_rel_pos[:,1],'r-',label="Deputy2")
plt.ylabel(r"$o$ Separation (m)")

plt.subplot(3,1,3)
plt.plot(times,d1_rel_pos[:,2],'b-',label="Deputy1")
plt.plot(times,d2_rel_pos[:,2],'r-',label="Deputy2")
plt.ylabel(r"$s$ Separation (m)")
plt.ylim((-1e-12,1e-12))
plt.xlabel("Time (s)")

#Save figure
plt.savefig('ECI_Base_0.svg', format='svg')

