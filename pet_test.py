from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits as orbits
from matplotlib.collections import LineCollection
from modules.Schweighart_J2_solved_clean import propagate_spacecraft
from modules.ECI_perturbations_abs import dX_dt
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
n_phases = 500
#Total evaluation points
n_times = int(n_orbits*n_phases)
times = np.linspace(0,ref.period*n_orbits,n_times) #Create list of times

#Initial reference orbit state
#pos_ref0,vel_ref0,LVLH0,Base0 = ref.ref_orbit_pos(0)

#Initial states of the satellites
chief_0 = orbits.init_chief(ref)
deputy1_0 = orbits.init_deputy(ref,1)
deputy2_0 = orbits.init_deputy(ref,2)

#------------------------------------------------------------------------------------------
### Schweighart solved version (see 2002 paper) #####
chief_p_states_sol = propagate_spacecraft(0,chief_0.to_Curvy().state,times,ref).transpose()
deputy1_p_states_sol = propagate_spacecraft(0,deputy1_0.to_Curvy().state,times,ref).transpose()
deputy2_p_states_sol = propagate_spacecraft(0,deputy2_0.to_Curvy().state,times,ref).transpose()

#------------------------------------------------------------------------------------------
### ECI version ###

#Tolerance and steps required for the integrator
rtol = 1e-12
atol = 1e-18
step = 1

def dX_dt2(t,state,ref):
    [x,y,z] = state[:3] #Position
    v = state[3:] #Velocity

    #First half of the differential vector (derivative of position, velocity)
    dX0 = v[0]
    dX1 = v[1]
    dX2 = v[2]

    J2 = 0.00108263 #J2 Parameter

    r = np.sqrt(x**2+y**2+z**2)
    #print(r-ref.R_orb)
    #Calculate J2 acceleration from the equation in ECI frame
    J2_fac1 = 3/2*J2*const.GM_earth.value*const.R_earth.value**2/r**5
    J2_fac2 = 5*z**2/r**2
    J2_p = J2_fac1*np.array([x*(J2_fac2-1),y*(J2_fac2-1),z*(J2_fac2-3)])

    r_hat = np.array([x,y,z])/r
    a = -const.GM_earth.value/r**2*r_hat + J2_p
    dX3 = a[0]
    dX4 = a[1]
    dX5 = a[2]
    return np.array([dX0,dX1,dX2,dX3,dX4,dX5])

#Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
X_d0 = solve_ivp(lambda t, y: dX_dt2(t,y,ref), [times[0],times[-1]], chief_0.state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X_d0.success:
    raise Exception("Integration failed!!!!")

#Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
X_d1 = solve_ivp(lambda t, y: dX_dt2(t,y,ref), [times[0],times[-1]], deputy1_0.state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X_d1.success:
    raise Exception("Integration failed!!!!")

X_d2 = solve_ivp(lambda t, y: dX_dt2(t,y,ref), [times[0],times[-1]], deputy2_0.state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
if not X_d2.success:
    raise Exception("Integration failed!!!!")

chief_p_states_eci = X_d0.y.transpose()
deputy1_p_states_eci = X_d1.y.transpose()
deputy2_p_states_eci = X_d2.y.transpose()

#-----------------------------------------------------------------------------------------
### Mike's adapted version of Schweighart ###

def J2_pet(sat0,ref):

    #DEFINE VARIABLES AS IN PAPER
    r_ref = ref.R_orb #Radius of the reference orbit (and chief)
    i_ref = ref.inc_0 #Inclination of the reference orbit (and chief)

    J2 = 0.00108263
    R_e = const.R_earth.value

    #Initial conditions
    [x_0,y_0,z_0] = sat0.pos
    dz_0 = sat0.vel[2]

    #Define variables
    c = ref.Sch_c
    n = ref.ang_vel
    k = ref.Sch_k
    h = 3*J2*R_e**2/(4*r_ref**2)

    #Equations of motion
    def J2_pet_func(t,state):
        [x,y,z] = state[:3] #Position
        [dx,dy,dz] = state[3:] #Velocity
        dX0 = dx
        dX1 = dy
        dX2 = dz

        theta = k*t
        dX3 = 2*n*c*dy + (5*c**2-2)*n**2*x + h*n**2*(12*np.sin(i_ref)**2*np.cos(2*theta)*x + 8*np.sin(i_ref)**2*np.sin(2*theta)*y+8*np.sin(2*i_ref)*np.sin(theta)*z)
        dX4 = -2*n*c*dx + h*n**2*(8*np.sin(i_ref)**2*np.sin(2*theta)*x + 7*np.sin(i_ref)**2*np.cos(2*theta)*y-2*np.sin(2*i_ref)*np.cos(theta)*z)
        dX5 = -(3*c**2-2)*n**2*z + h*n**2*(8*np.sin(2*i_ref)*np.sin(theta)*x - 2*np.sin(2*i_ref)*np.cos(theta)*y-5*np.sin(i_ref)**2*np.cos(2*theta)*z)
        return np.array([dX0,dX1,dX2,dX3,dX4,dX5])

    return J2_pet_func

#Equations of motion
J2_func0 = J2_pet(chief_0.to_Curvy(),ref)
J2_func1 = J2_pet(deputy1_0.to_Curvy(),ref)
J2_func2 = J2_pet(deputy2_0.to_Curvy(),ref)

X2_d0 = solve_ivp(J2_func0, [times[0],times[-1]], chief_0.to_Curvy().state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X2_d0.success:
    raise Exception("Integration failed!!!!")

X2_d1 = solve_ivp(J2_func1, [times[0],times[-1]], deputy1_0.to_Curvy().state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X2_d1.success:
    raise Exception("Integration failed!!!!")

X2_d2 = solve_ivp(J2_func2, [times[0],times[-1]], deputy2_0.to_Curvy().state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
if not X2_d2.success:
    raise Exception("Integration failed!!!!")

chief_p_states_num = X2_d0.y.transpose()
deputy1_p_states_num = X2_d1.y.transpose()
deputy2_p_states_num = X2_d2.y.transpose()

#------------------------------------------------------------------------------------------
### Old version (LVLH Matrix) ###

rtol = 1e-6
atol = 1e-9
step = 10000

#Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
X3_c = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], chief_0.to_LVLH().state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X3_c.success:
    raise Exception("Integration failed!!!!")

X3_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], deputy1_0.to_LVLH().state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X3_d1.success:
    raise Exception("Integration failed!!!!")

X3_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], deputy2_0.to_LVLH().state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
if not X3_d2.success:
    raise Exception("Integration failed!!!!")

chief_p_states_bad = X3_c.y.transpose()
deputy1_p_states_bad = X3_d1.y.transpose()
deputy2_p_states_bad = X3_d2.y.transpose()

#------------------------------------------------------------------------------------------

ECI_rc = np.zeros((len(times),3))
c_eci = []
d1_eci = []
d2_eci = []
c_num = []
d1_num = []
d2_num = []
c_bad = []
d1_bad = []
d2_bad = []
c_sol = []
d1_sol = []
d2_sol = []

print("Integration Done")
for i in range(len(times)):
    pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(times[i],True)
    c_eci.append(orbits.ECI_Sat(chief_p_states_eci[i,:3],chief_p_states_eci[i,3:],times[i],ref).to_LVLH())
    d1_eci.append(orbits.ECI_Sat(deputy1_p_states_eci[i,:3],deputy1_p_states_eci[i,3:],times[i],ref).to_LVLH())
    d2_eci.append(orbits.ECI_Sat(deputy2_p_states_eci[i,:3],deputy2_p_states_eci[i,3:],times[i],ref).to_LVLH())
    c_num.append(orbits.Curvy_Sat(chief_p_states_num[i,:3],chief_p_states_num[i,3:],times[i],ref).to_LVLH())
    d1_num.append(orbits.Curvy_Sat(deputy1_p_states_num[i,:3],deputy1_p_states_num[i,3:],times[i],ref).to_LVLH())
    d2_num.append(orbits.Curvy_Sat(deputy2_p_states_num[i,:3],deputy2_p_states_num[i,3:],times[i],ref).to_LVLH())
    c_sol.append(orbits.Curvy_Sat(chief_p_states_sol[i,:3],chief_p_states_sol[i,3:],times[i],ref).to_LVLH())
    d1_sol.append(orbits.Curvy_Sat(deputy1_p_states_sol[i,:3],deputy1_p_states_sol[i,3:],times[i],ref).to_LVLH())
    d2_sol.append(orbits.Curvy_Sat(deputy2_p_states_sol[i,:3],deputy2_p_states_sol[i,3:],times[i],ref).to_LVLH())
    c_bad.append(orbits.LVLH_Sat(chief_p_states_bad[i,:3],chief_p_states_bad[i,3:],times[i],ref))
    d1_bad.append(orbits.LVLH_Sat(deputy1_p_states_bad[i,:3],deputy1_p_states_bad[i,3:],times[i],ref))
    d2_bad.append(orbits.LVLH_Sat(deputy2_p_states_bad[i,:3],deputy2_p_states_bad[i,3:],times[i],ref))
print("Classifying Done")

#--------------------------------------------------------------------------------------------- #
#Separations and accelerations
rho_d1_eci = np.zeros(n_times)
rho_d2_eci = np.zeros(n_times)
rho_d1_num = np.zeros(n_times)
rho_d2_num = np.zeros(n_times)
rho_d1_sol = np.zeros(n_times)
rho_d2_sol = np.zeros(n_times)
rho_d1_bad = np.zeros(n_times)
rho_d2_bad = np.zeros(n_times)
xi_d1_eci = np.zeros(n_times)
xi_d2_eci = np.zeros(n_times)
xi_d1_num = np.zeros(n_times)
xi_d2_num = np.zeros(n_times)
xi_d1_sol = np.zeros(n_times)
xi_d2_sol = np.zeros(n_times)
xi_d1_bad = np.zeros(n_times)
xi_d2_bad = np.zeros(n_times)
eta_d1_eci = np.zeros(n_times)
eta_d2_eci = np.zeros(n_times)
eta_d1_num = np.zeros(n_times)
eta_d2_num = np.zeros(n_times)
eta_d1_sol = np.zeros(n_times)
eta_d2_sol = np.zeros(n_times)
eta_d1_bad = np.zeros(n_times)
eta_d2_bad = np.zeros(n_times)

for ix in range(n_times):
    #Component of perturbed orbit in rho direction
    rho_d1_eci[ix] = d1_eci[ix].pos[0] - c_eci[ix].pos[0]
    rho_d2_eci[ix] = d2_eci[ix].pos[0] - c_eci[ix].pos[0]
    rho_d1_num[ix] = d1_num[ix].pos[0] - c_num[ix].pos[0]
    rho_d2_num[ix] = d2_num[ix].pos[0] - c_num[ix].pos[0]
    rho_d1_sol[ix] = d1_sol[ix].pos[0] - c_sol[ix].pos[0]
    rho_d2_sol[ix] = d2_sol[ix].pos[0] - c_sol[ix].pos[0]
    rho_d1_bad[ix] = d1_bad[ix].pos[0] - c_bad[ix].pos[0]
    rho_d2_bad[ix] = d2_bad[ix].pos[0] - c_bad[ix].pos[0]

    #Component of perturbed orbit in xi direction
    xi_d1_eci[ix] = d1_eci[ix].pos[1] - c_eci[ix].pos[1]
    xi_d2_eci[ix] = d2_eci[ix].pos[1] - c_eci[ix].pos[1]
    xi_d1_num[ix] = d1_num[ix].pos[1] - c_num[ix].pos[1]
    xi_d2_num[ix] = d2_num[ix].pos[1] - c_num[ix].pos[1]
    xi_d1_sol[ix] = d1_sol[ix].pos[1] - c_sol[ix].pos[1]
    xi_d2_sol[ix] = d2_sol[ix].pos[1] - c_sol[ix].pos[1]
    xi_d1_bad[ix] = d1_bad[ix].pos[1] - c_bad[ix].pos[1]
    xi_d2_bad[ix] = d2_bad[ix].pos[1] - c_bad[ix].pos[1]

    #Component of perturbed orbit in eta direction
    eta_d1_eci[ix] = d1_eci[ix].pos[2] - c_eci[ix].pos[2]
    eta_d2_eci[ix] = d2_eci[ix].pos[2] - c_eci[ix].pos[2]
    eta_d1_num[ix] = d1_num[ix].pos[2] - c_num[ix].pos[2]
    eta_d2_num[ix] = d2_num[ix].pos[2] - c_num[ix].pos[2]
    eta_d1_sol[ix] = d1_sol[ix].pos[2] - c_sol[ix].pos[2]
    eta_d2_sol[ix] = d2_sol[ix].pos[2] - c_sol[ix].pos[2]
    eta_d1_bad[ix] = d1_bad[ix].pos[2] - c_bad[ix].pos[2]
    eta_d2_bad[ix] = d2_bad[ix].pos[2] - c_bad[ix].pos[2]

# ---------------------------------------------------------------------- #

#Plot separation along the rho direction
plt.figure(1)
plt.clf()
plt.subplot(3,3,1)
plt.plot(times,rho_d1_eci,"b-",label="Deputy 1, ECI")
plt.plot(times,rho_d2_eci,"r-",label="Deputy 2, ECI")
plt.plot(times,rho_d1_bad,"b--",label="Deputy 1, Old")
plt.plot(times,rho_d2_bad,"r--",label="Deputy 2, Old")
plt.plot(times,rho_d1_eci-rho_d1_bad,"c--",label="Deputy 1, Old Residuals")
plt.plot(times,rho_d2_eci-rho_d2_bad,"m--",label="Deputy 2, Old Residuals")
plt.ylabel("Rho Separation(m)")
plt.title("Old Integration")
plt.legend()

plt.subplot(3,3,2)
plt.plot(times,rho_d1_eci,"b-",label="Deputy 1, ECI")
plt.plot(times,rho_d2_eci,"r-",label="Deputy 2, ECI")
plt.plot(times,rho_d1_num,"b--",label="Deputy 1, Mike")
plt.plot(times,rho_d2_num,"r--",label="Deputy 2, Mike")
plt.plot(times,rho_d1_eci-rho_d1_num,"c--",label="Deputy 1, Mike Residuals")
plt.plot(times,rho_d2_eci-rho_d2_num,"m--",label="Deputy 2, Mike Residuals")
plt.title("Mike's Adapted Schweighart")
plt.legend()

plt.subplot(3,3,3)
plt.plot(times,rho_d1_eci,"b-",label="Deputy 1, ECI")
plt.plot(times,rho_d2_eci,"r-",label="Deputy 2, ECI")
plt.plot(times,rho_d1_sol,"b--",label="Deputy 1, Schweighart")
plt.plot(times,rho_d2_sol,"r--",label="Deputy 2, Schweighart")
plt.plot(times,rho_d1_eci-rho_d1_sol,"c--",label="Deputy 1, Schweighart Residuals")
plt.plot(times,rho_d2_eci-rho_d2_sol,"m--",label="Deputy 2, Schweighart Residuals")
plt.title("Schweighart's version")
plt.legend()

plt.subplot(3,3,4)
plt.plot(times,xi_d1_eci,"b-",label="Deputy 1, ECI")
plt.plot(times,xi_d2_eci,"r-",label="Deputy 2, ECI")
plt.plot(times,xi_d1_bad,"b--",label="Deputy 1, Old")
plt.plot(times,xi_d2_bad,"r--",label="Deputy 2, Old")
plt.plot(times,xi_d1_eci-xi_d1_bad,"c--",label="Deputy 1, Old Residuals")
plt.plot(times,xi_d2_eci-xi_d2_bad,"m--",label="Deputy 2, Old Residuals")
plt.ylabel("Xi Separation(m)")
plt.legend()

plt.subplot(3,3,5)
plt.plot(times,xi_d1_eci,"b-",label="Deputy 1, ECI")
plt.plot(times,xi_d2_eci,"r-",label="Deputy 2, ECI")
plt.plot(times,xi_d1_num,"b--",label="Deputy 1, Mike")
plt.plot(times,xi_d2_num,"r--",label="Deputy 2, Mike")
plt.plot(times,xi_d1_eci-xi_d1_num,"c--",label="Deputy 1, Mike Residuals")
plt.plot(times,xi_d2_eci-xi_d2_num,"m--",label="Deputy 2, Mike Residuals")
plt.legend()

plt.subplot(3,3,6)
plt.plot(times,xi_d1_eci,"b-",label="Deputy 1, ECI")
plt.plot(times,xi_d2_eci,"r-",label="Deputy 2, ECI")
plt.plot(times,xi_d1_sol,"b--",label="Deputy 1, Schweighart")
plt.plot(times,xi_d2_sol,"r--",label="Deputy 2, Schweighart")
plt.plot(times,xi_d1_eci-xi_d1_sol,"c--",label="Deputy 1, Schweighart Residuals")
plt.plot(times,xi_d2_eci-xi_d2_sol,"m--",label="Deputy 2, Schweighart Residuals")
plt.legend()

plt.subplot(3,3,7)
plt.plot(times,eta_d1_eci,"b-",label="Deputy 1, ECI")
plt.plot(times,eta_d2_eci,"r-",label="Deputy 2, ECI")
plt.plot(times,eta_d1_bad,"b--",label="Deputy 1, Old")
plt.plot(times,eta_d2_bad,"r--",label="Deputy 2, Old")
plt.plot(times,eta_d1_eci-eta_d1_bad,"c--",label="Deputy 1, Old Residuals")
plt.plot(times,eta_d2_eci-eta_d2_bad,"m--",label="Deputy 2, Old Residuals")
plt.ylabel("Eta Separation(m)")
plt.xlabel("Times(s)")
plt.legend()

plt.subplot(3,3,8)
plt.plot(times,eta_d1_eci,"b-",label="Deputy 1, ECI")
plt.plot(times,eta_d2_eci,"r-",label="Deputy 2, ECI")
plt.plot(times,eta_d1_num,"b--",label="Deputy 1, Mike")
plt.plot(times,eta_d2_num,"r--",label="Deputy 2, Mike")
plt.plot(times,eta_d1_eci-eta_d1_num,"c--",label="Deputy 1, Mike Residuals")
plt.plot(times,eta_d2_eci-eta_d2_num,"m--",label="Deputy 2, Mike Residuals")
plt.xlabel("Times(s)")
plt.legend()

plt.subplot(3,3,9)
plt.plot(times,eta_d1_eci,"b-",label="Deputy 1, ECI")
plt.plot(times,eta_d2_eci,"r-",label="Deputy 2, ECI")
plt.plot(times,eta_d1_sol,"b--",label="Deputy 1, Schweighart")
plt.plot(times,eta_d2_sol,"r--",label="Deputy 2, Schweighart")
plt.plot(times,eta_d1_eci-eta_d1_sol,"c--",label="Deputy 1, Schweighart Residuals")
plt.plot(times,eta_d2_eci-eta_d2_sol,"m--",label="Deputy 2, Schweighart Residuals")
plt.xlabel("Times(s)")
plt.legend()

plt.suptitle('Separations against time due to perturbations')


plt.figure(2)
plt.clf()
plt.subplot(3,3,1)
plt.plot(times,rho_d1_eci-rho_d1_bad,"c--",label="Deputy 1, Old Residuals")
plt.plot(times,rho_d2_eci-rho_d2_bad,"m--",label="Deputy 2, Old Residuals")
plt.ylabel("Rho Separation(m)")
plt.title("Old Integration")
plt.legend()

plt.subplot(3,3,2)
plt.plot(times,rho_d1_eci-rho_d1_num,"c--",label="Deputy 1, Mike Residuals")
plt.plot(times,rho_d2_eci-rho_d2_num,"m--",label="Deputy 2, Mike Residuals")
plt.title("Mike's Adapted Schweighart")
plt.legend()

plt.subplot(3,3,3)
plt.plot(times,rho_d1_eci-rho_d1_sol,"c--",label="Deputy 1, Schweighart Residuals")
plt.plot(times,rho_d2_eci-rho_d2_sol,"m--",label="Deputy 2, Schweighart Residuals")
plt.title("Schweighart's version")
plt.legend()

plt.subplot(3,3,4)
plt.plot(times,xi_d1_eci-xi_d1_bad,"c--",label="Deputy 1, Old Residuals")
plt.plot(times,xi_d2_eci-xi_d2_bad,"m--",label="Deputy 2, Old Residuals")
plt.ylabel("Xi Separation(m)")
plt.legend()

plt.subplot(3,3,5)
plt.plot(times,xi_d1_eci-xi_d1_num,"c--",label="Deputy 1, Mike Residuals")
plt.plot(times,xi_d2_eci-xi_d2_num,"m--",label="Deputy 2, Mike Residuals")
plt.legend()

plt.subplot(3,3,6)
plt.plot(times,xi_d1_eci-xi_d1_sol,"c--",label="Deputy 1, Schweighart Residuals")
plt.plot(times,xi_d2_eci-xi_d2_sol,"m--",label="Deputy 2, Schweighart Residuals")
plt.legend()

plt.subplot(3,3,7)
plt.plot(times,eta_d1_eci-eta_d1_bad,"c--",label="Deputy 1, Old Residuals")
plt.plot(times,eta_d2_eci-eta_d2_bad,"m--",label="Deputy 2, Old Residuals")
plt.ylabel("Eta Separation(m)")
plt.xlabel("Times(s)")
plt.legend()

plt.subplot(3,3,8)
plt.plot(times,eta_d1_eci-eta_d1_num,"c--",label="Deputy 1, Mike Residuals")
plt.plot(times,eta_d2_eci-eta_d2_num,"m--",label="Deputy 2, Mike Residuals")
plt.xlabel("Times(s)")
plt.legend()

plt.subplot(3,3,9)
plt.plot(times,eta_d1_eci-eta_d1_sol,"c--",label="Deputy 1, Schweighart Residuals")
plt.plot(times,eta_d2_eci-eta_d2_sol,"m--",label="Deputy 2, Schweighart Residuals")
plt.xlabel("Times(s)")
plt.legend()

plt.suptitle('Residual Separations against time due to perturbations')

