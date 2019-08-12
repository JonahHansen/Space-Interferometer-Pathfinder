from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits as orbits
from matplotlib.collections import LineCollection
from modules.Schweighart_J2_solved_clean import propagate_spacecraft
from modules.Schweighart_J2 import J2_pet

plt.ion()

alt = 500e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(89) #20
#Longitude of the Ascending Node
Om_0 = np.radians(0) #0

#Stellar vector
ra = np.radians(45) #90
dec = np.radians(56)#-40

#The max distance to the other satellites in m
delta_r_max = 0.3e3

#------------------------------------------------------------------------------------------
#Calculate reference orbit, in the geocentric (ECI) frame (See Orbit module)
ref = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

#Number of orbits
n_orbits = 1
#Number of phases in each orbit
n_phases = 500
#Total evaluation points
n_times = int(n_orbits*n_phases)
times = np.linspace(0,ref.period*n_orbits,n_times) #Create list of times

#Initial reference orbit state
pos_ref0,vel_ref0,LVLH0,Base0 = ref.ref_orbit_pos(0)

#Initial states of the satellites
chief_0 = orbits.init_chief(ref,0)
deputy1_0 = orbits.init_deputy(ref,0,1)
deputy2_0 = orbits.init_deputy(ref,0,2)

#------------------------------------------------------------------------------------------
### SOLVED #####
chief_p_states_sol = propagate_spacecraft(0,chief_0.to_LVLH(pos_ref0,vel_ref0,LVLH0).state,times,ref).transpose()
deputy1_p_states_sol = propagate_spacecraft(0,deputy1_0.to_LVLH(pos_ref0,vel_ref0,LVLH0).state,times,ref).transpose()
deputy2_p_states_sol = propagate_spacecraft(0,deputy2_0.to_LVLH(pos_ref0,vel_ref0,LVLH0).state,times,ref).transpose()

d1_rel_sol = deputy1_p_states_sol - chief_p_states_sol
d2_rel_sol = deputy2_p_states_sol - chief_p_states_sol

#------------------------------------------------------------------------------------------

#Tolerance and steps required for the integrator
rtol = 1e-9
atol = 1e-18
step = 10

def dX_dt(t,state,ref):
    [x,y,z] = state[:3] #Position
    v = state[3:] #Velocity

    #First half of the differential vector (derivative of position, velocity)
    dX0 = v[0]
    dX1 = v[1]
    dX2 = v[2]

    J2 = 0.00108263 #J2 Parameter

    #Calculate J2 acceleration from the equation in ECI frame
    J2_fac1 = 3/2*J2*const.GM_earth.value*const.R_earth.value**2/ref.R_orb**5
    J2_fac2 = 5*z**2/ref.R_orb**2
    J2_p = J2_fac1*np.array([x*(J2_fac2-1),y*(J2_fac2-1),z*(J2_fac2-3)])

    r_hat = np.array([x,y,z])/np.linalg.norm(np.sqrt(x**2+y**2+z**2))
    a = -const.GM_earth.value/ref.R_orb**2*r_hat + J2_p
    dX3 = a[0]
    dX4 = a[1]
    dX5 = a[2]
    return np.array([dX0,dX1,dX2,dX3,dX4,dX5])

#Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
X_d0 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], chief_0.state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X_d0.success:
    raise Exception("Integration failed!!!!")

#Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], deputy1_0.state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X_d1.success:
    raise Exception("Integration failed!!!!")

X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], deputy2_0.state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
if not X_d2.success:
    raise Exception("Integration failed!!!!")

chief_p_states_eci = X_d0.y.transpose()
deputy1_p_states_eci = X_d1.y.transpose()
deputy2_p_states_eci = X_d2.y.transpose()

d1_rel_eci = deputy1_p_states_eci - chief_p_states_eci
d2_rel_eci = deputy2_p_states_eci - chief_p_states_eci

#------------------------------------------------------------------------------------------

#Equations of motion
J2_func0 = J2_pet(chief_0.to_LVLH(pos_ref0,vel_ref0,LVLH0),ref)
J2_func1 = J2_pet(deputy1_0.to_LVLH(pos_ref0,vel_ref0,LVLH0),ref)
J2_func2 = J2_pet(deputy2_0.to_LVLH(pos_ref0,vel_ref0,LVLH0),ref)

X2_d0 = solve_ivp(J2_func0, [times[0],times[-1]], chief_0.to_LVLH(pos_ref0,vel_ref0,LVLH0).state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X2_d0.success:
    raise Exception("Integration failed!!!!")

X2_d1 = solve_ivp(J2_func1, [times[0],times[-1]], deputy1_0.to_LVLH(pos_ref0,vel_ref0,LVLH0).state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X2_d1.success:
    raise Exception("Integration failed!!!!")

X2_d2 = solve_ivp(J2_func2, [times[0],times[-1]], deputy2_0.to_LVLH(pos_ref0,vel_ref0,LVLH0).state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
if not X2_d2.success:
    raise Exception("Integration failed!!!!")

chief_p_states_num = X2_d0.y.transpose()
deputy1_p_states_num = X2_d1.y.transpose()
deputy2_p_states_num = X2_d2.y.transpose()

d1_rel_num = deputy1_p_states_num - chief_p_states_num
d2_rel_num = deputy2_p_states_num - chief_p_states_num

#------------------------------------------------------------------------------------------

ECI_rc = np.zeros((len(times),3))
c_eci_lvlh = []
d1_eci_lvlh = []
d2_eci_lvlh = []
d1_relsat_num = []
d2_relsat_num = []
d1_relsat_sol = []
d2_relsat_sol = []

print("Integration Done")
for i in range(len(times)):
    pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(times[i],True)
    c_eci_lvlh.append(orbits.ECI_Sat(chief_p_states_eci[i,:3],chief_p_states_eci[i,3:],times[i],ref).to_LVLH(pos_ref,vel_ref,LVLH))
    d1_eci_lvlh.append(orbits.ECI_Sat(deputy1_p_states_eci[i,:3],deputy1_p_states_eci[i,3:],times[i],ref).to_LVLH(pos_ref,vel_ref,LVLH))
    d2_eci_lvlh.append(orbits.ECI_Sat(deputy2_p_states_eci[i,:3],deputy2_p_states_eci[i,3:],times[i],ref).to_LVLH(pos_ref,vel_ref,LVLH))
    d1_relsat_num.append(orbits.LVLH_Sat(d1_rel_num[i,:3],d1_rel_num[i,3:],times[i],ref))
    d2_relsat_num.append(orbits.LVLH_Sat(d2_rel_num[i,:3],d2_rel_num[i,3:],times[i],ref))
    d1_relsat_sol.append(orbits.LVLH_Sat(d1_rel_sol[i,:3],d1_rel_sol[i,3:],times[i],ref))
    d2_relsat_sol.append(orbits.LVLH_Sat(d2_rel_sol[i,:3],d2_rel_sol[i,3:],times[i],ref))
print("Classifying Done")

#--------------------------------------------------------------------------------------------- #
#Separations and accelerations
rho_d1_eci = np.zeros(n_times)
rho_d2_eci = np.zeros(n_times)
rho_d1_num = np.zeros(n_times)
rho_d2_num = np.zeros(n_times)
rho_d1_sol = np.zeros(n_times)
rho_d2_sol = np.zeros(n_times)
xi_d1_eci = np.zeros(n_times)
xi_d2_eci = np.zeros(n_times)
xi_d1_num = np.zeros(n_times)
xi_d2_num = np.zeros(n_times)
xi_d1_sol = np.zeros(n_times)
xi_d2_sol = np.zeros(n_times)
eta_d1_eci = np.zeros(n_times)
eta_d2_eci = np.zeros(n_times)
eta_d1_num = np.zeros(n_times)
eta_d2_num = np.zeros(n_times)
eta_d1_sol = np.zeros(n_times)
eta_d2_sol = np.zeros(n_times)

for ix in range(n_times):
    #Component of perturbed orbit in rho direction
    rho_d1_eci[ix] = d1_eci_lvlh[ix].pos[0] - c_eci_lvlh[ix].pos[0]
    rho_d2_eci[ix] = d2_eci_lvlh[ix].pos[0] - c_eci_lvlh[ix].pos[0]
    rho_d1_num[ix] = d1_relsat_num[ix].pos[0]
    rho_d2_num[ix] = d2_relsat_num[ix].pos[0]
    rho_d1_sol[ix] = d1_relsat_sol[ix].pos[0]
    rho_d2_sol[ix] = d2_relsat_sol[ix].pos[0]

    #Component of perturbed orbit in xi direction
    xi_d1_eci[ix] = d1_eci_lvlh[ix].pos[1] - c_eci_lvlh[ix].pos[1]
    xi_d2_eci[ix] = d2_eci_lvlh[ix].pos[1] - c_eci_lvlh[ix].pos[1]
    xi_d1_num[ix] = d1_relsat_num[ix].pos[1]
    xi_d2_num[ix] = d2_relsat_num[ix].pos[1]
    xi_d1_sol[ix] = d1_relsat_sol[ix].pos[1]
    xi_d2_sol[ix] = d2_relsat_sol[ix].pos[1]

    #Component of perturbed orbit in eta direction
    eta_d1_eci[ix] = d1_eci_lvlh[ix].pos[2] - c_eci_lvlh[ix].pos[2]
    eta_d2_eci[ix] = d2_eci_lvlh[ix].pos[2] - c_eci_lvlh[ix].pos[2]
    eta_d1_num[ix] = d1_relsat_num[ix].pos[2]
    eta_d2_num[ix] = d2_relsat_num[ix].pos[2]
    eta_d1_sol[ix] = d1_relsat_sol[ix].pos[2]
    eta_d2_sol[ix] = d2_relsat_sol[ix].pos[2]

# ---------------------------------------------------------------------- #

#Plot separation along the rho direction
plt.figure(1)
plt.clf()
#plt.plot(times,rho_d1_eci,"b--",label="Deputy 1, ECI")
#plt.plot(times,rho_d2_eci,"r--",label="Deputy 2, ECI")
#plt.plot(times,rho_d1_num,"b:",label="Deputy 1, Numerical")
#plt.plot(times,rho_d2_num,"r:",label="Deputy 2, Numerical")
plt.plot(times,rho_d1_sol,"b-",label="Deputy 1, Solved")
plt.plot(times,rho_d2_sol,"r-",label="Deputy 2, Solved")
#plt.plot(times,rho_d1_sol-rho_d1_num,"c:",label="Deputy 1, Num Residuals")
#plt.plot(times,rho_d2_sol-rho_d2_num,"m:",label="Deputy 2, Num Residuals")
#plt.plot(times,rho_d1_sol-rho_d1_eci,"c--",label="Deputy 1, ECI Residuals")
#plt.plot(times,rho_d2_sol-rho_d2_eci,"m--",label="Deputy 2, ECI Residuals")
plt.xlabel("Times(s)")
plt.ylabel("Rho Separation(m)")
plt.title('Rho Separations against time due to perturbations')
plt.legend()

#Plot separation along the xi direction
plt.figure(2)
plt.clf()
plt.plot(times,xi_d1_eci,"b--",label="Deputy 1, ECI")
plt.plot(times,xi_d2_eci,"r--",label="Deputy 2, ECI")
plt.plot(times,xi_d1_num,"b:",label="Deputy 1, Numerical")
plt.plot(times,xi_d2_num,"r:",label="Deputy 2, Numerical")
plt.plot(times,xi_d1_sol,"b-",label="Deputy 1, Solved")
plt.plot(times,xi_d2_sol,"r-",label="Deputy 2, Solved")
plt.plot(times,xi_d1_sol-xi_d1_num,"c:",label="Deputy 1, Num Residuals")
plt.plot(times,xi_d2_sol-xi_d2_num,"m:",label="Deputy 2, Num Residuals")
plt.plot(times,xi_d1_sol-xi_d1_eci,"c--",label="Deputy 1, ECI Residuals")
plt.plot(times,xi_d2_sol-xi_d2_eci,"m--",label="Deputy 2, ECI Residuals")
plt.xlabel("Times(s)")
plt.ylabel("Xi Separation(m)")
plt.title('Xi Separations against time due to perturbations')
plt.legend()

#Plot separation along the eta direction
plt.figure(3)
plt.clf()
plt.plot(times,eta_d1_eci,"b--",label="Deputy 1, ECI")
plt.plot(times,eta_d2_eci,"r--",label="Deputy 2, ECI")
plt.plot(times,eta_d1_num,"b:",label="Deputy 1, Numerical")
plt.plot(times,eta_d2_num,"r:",label="Deputy 2, Numerical")
plt.plot(times,eta_d1_sol,"b-",label="Deputy 1, Solved")
plt.plot(times,eta_d2_sol,"r-",label="Deputy 2, Solved")
plt.plot(times,eta_d1_sol-eta_d1_num,"c:",label="Deputy 1, Num Residuals")
plt.plot(times,eta_d2_sol-eta_d2_num,"m:",label="Deputy 2, Num Residuals")
plt.plot(times,eta_d1_sol-eta_d1_eci,"c--",label="Deputy 1, ECI Residuals")
plt.plot(times,eta_d2_sol-eta_d2_eci,"m--",label="Deputy 2, ECI Residuals")
plt.xlabel("Times(s)")
plt.ylabel("Eta Separation(m)")
plt.title('Eta Separations against time due to perturbations')
plt.legend()

