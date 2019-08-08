from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits as orbits
from matplotlib.collections import LineCollection
from modules.Schweighart_J2_solved_clean import propagate_spacecraft
from modules.ECI_perturbations import dX_dt

plt.ion()

alt = 500e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(34) #20
#Longitude of the Ascending Node
Om_0 = np.radians(0) #0

#Stellar vector
ra = np.radians(0) #90
dec = np.radians(45)#-40

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
pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(0)

#Initial states of the satellites
chief_0 = orbits.init_chief(ref,0).to_LVLH(pos_ref,vel_ref,LVLH)
deputy1_0 = orbits.init_deputy(ref,0,1).to_LVLH(pos_ref,vel_ref,LVLH)
deputy2_0 = orbits.init_deputy(ref,0,2).to_LVLH(pos_ref,vel_ref,LVLH)

chief_p_states = propagate_spacecraft(0,chief_0.state,times,ref).transpose()
deputy1_p_states = propagate_spacecraft(0,deputy1_0.state,times,ref).transpose()
deputy2_p_states = propagate_spacecraft(0,deputy2_0.state,times,ref).transpose()

d1_rel_sch = deputy1_p_states - chief_p_states
d2_rel_sch = deputy2_p_states - chief_p_states

#Tolerance and steps required for the integrator
rtol = 1e-9
atol = 1e-18
step = 10

#Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], deputy1_0.state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X_d1.success:
    raise Exception("Integration failed!!!!")

X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], deputy2_0.state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
if not X_d2.success:
    raise Exception("Integration failed!!!!")

d1_rel_eci = X_d1.y.transpose()
d2_rel_eci = X_d2.y.transpose()


ECI_rc = np.zeros((len(times),3))
d1_relsat_eci = []
d2_relsat_eci = []
d1_relsat_sch = []
d2_relsat_sch = []

print("Integration Done")
for i in range(len(times)):
    pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(times[i],True)
    d1_relsat_eci.append(orbits.LVLH_Sat(d1_rel_eci[i,:3],d1_rel_eci[i,3:],times[i],ref))
    d2_relsat_eci.append(orbits.LVLH_Sat(d2_rel_eci[i,:3],d2_rel_eci[i,3:],times[i],ref))
    d1_relsat_sch.append(orbits.LVLH_Sat(d1_rel_sch[i,:3],d1_rel_sch[i,3:],times[i],ref))
    d2_relsat_sch.append(orbits.LVLH_Sat(d2_rel_sch[i,:3],d2_rel_sch[i,3:],times[i],ref))
print("Classifying Done")

#--------------------------------------------------------------------------------------------- #
#Separations and accelerations
rho_d1_eci = np.zeros(n_times)
rho_d2_eci = np.zeros(n_times)
rho_d1_sch = np.zeros(n_times)
rho_d2_sch = np.zeros(n_times)
xi_d1_eci = np.zeros(n_times)
xi_d2_eci = np.zeros(n_times)
xi_d1_sch = np.zeros(n_times)
xi_d2_sch = np.zeros(n_times)
eta_d1_eci = np.zeros(n_times)
eta_d2_eci = np.zeros(n_times)
eta_d1_sch = np.zeros(n_times)
eta_d2_sch = np.zeros(n_times)


for ix in range(n_times):
    #Component of perturbed orbit in rho direction
    rho_d1_eci[ix] = d1_relsat_eci[ix].pos[0]
    rho_d2_eci[ix] = d2_relsat_eci[ix].pos[0]
    rho_d1_sch[ix] = d1_relsat_sch[ix].pos[0]
    rho_d2_sch[ix] = d2_relsat_sch[ix].pos[0]

    #Component of perturbed orbit in xi direction
    xi_d1_eci[ix] = d1_relsat_eci[ix].pos[1]
    xi_d2_eci[ix] = d2_relsat_eci[ix].pos[1]
    xi_d1_sch[ix] = d1_relsat_sch[ix].pos[1]
    xi_d2_sch[ix] = d2_relsat_sch[ix].pos[1]

    #Component of perturbed orbit in eta direction
    eta_d1_eci[ix] = d1_relsat_eci[ix].pos[2]
    eta_d2_eci[ix] = d2_relsat_eci[ix].pos[2]
    eta_d1_sch[ix] = d1_relsat_sch[ix].pos[2]
    eta_d2_sch[ix] = d2_relsat_sch[ix].pos[2]

# ---------------------------------------------------------------------- #

#Plot separation along the rho direction
plt.figure(1)
plt.clf()
plt.plot(times,rho_d1_eci,"b-",label="Deputy 1, ECI")
plt.plot(times,rho_d2_eci,"r-",label="Deputy 2, ECI")
plt.plot(times,rho_d1_sch,"b--",label="Deputy 1, Schweighart")
plt.plot(times,rho_d2_sch,"r--",label="Deputy 2, Schweighart")
plt.plot(times,rho_d1_eci-rho_d1_sch,"c-",label="Deputy 1, Residuals")
plt.plot(times,rho_d2_eci-rho_d2_sch,"m-",label="Deputy 2, Residuals")
plt.xlabel("Times(s)")
plt.ylabel("Rho Separation(m)")
plt.title('Rho Separations against time due to perturbations')
plt.legend()

#Plot separation along the xi direction
plt.figure(2)
plt.clf()
plt.plot(times,xi_d1_eci,"b-",label="Deputy 1, ECI")
plt.plot(times,xi_d2_eci,"r-",label="Deputy 2, ECI")
plt.plot(times,xi_d1_sch,"b--",label="Deputy 1, Schweighart")
plt.plot(times,xi_d2_sch,"r--",label="Deputy 2, Schweighart")
plt.plot(times,xi_d1_eci-xi_d1_sch,"c-",label="Deputy 1, Residuals")
plt.plot(times,xi_d2_eci-xi_d2_sch,"m-",label="Deputy 2, Residuals")
plt.xlabel("Times(s)")
plt.ylabel("Xi Separation(m)")
plt.title('Xi Separations against time due to perturbations')
plt.legend()

#Plot separation along the eta direction
plt.figure(3)
plt.clf()
plt.plot(times,eta_d1_eci,"b-",label="Deputy 1, ECI")
plt.plot(times,eta_d2_eci,"r-",label="Deputy 2, ECI")
plt.plot(times,eta_d1_sch,"b--",label="Deputy 1, Schweighart")
plt.plot(times,eta_d2_sch,"r--",label="Deputy 2, Schweighart")
plt.plot(times,eta_d1_eci-eta_d1_sch,"c-",label="Deputy 1, Residuals")
plt.plot(times,eta_d2_eci-eta_d2_sch,"m-",label="Deputy 2, Residuals")
plt.xlabel("Times(s)")
plt.ylabel("Eta Separation(m)")
plt.title('Eta Separations against time due to perturbations')
plt.legend()

