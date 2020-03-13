""" Script comparing a bunch of methods to calculate J2 perturbed satellite motion """
""" None really work, and this is a mess of a script """

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits as orbits
from scipy.optimize import fsolve
from matplotlib.collections import LineCollection
from modules.Analytical_LVLH_motion import propagate_spacecraft
from modules.old_perturbations import dX_dt
from modules.Numerical_LVLH_motion import J2_pert_num
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
chief_p_states_sol = propagate_spacecraft(0,chief_0.to_Curvy(ref_orbit=True).state,times,ref).transpose()
deputy1_p_states_sol = propagate_spacecraft(0,deputy1_0.to_Curvy(ref_orbit=True).state,times,ref).transpose()# - chief_p_states_sol
deputy2_p_states_sol = propagate_spacecraft(0,deputy2_0.to_Curvy(ref_orbit=True).state,times,ref).transpose()# - chief_p_states_sol

print("Done Solved")
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

print("Done ECI")
#-----------------------------------------------------------------------------------------
### Mike's adapted version of Schweighart ###

def J2_pert_Mike(sat0,ref):

    #DEFINE VARIABLES AS IN PAPER
    r_ref = ref.R_orb #Radius of the reference orbit (and chief)


    J2 = 0.00108263
    R_e = const.R_earth.value

    #Initial conditions
    [x_0,y_0,z_0] = sat0.pos
    dz_0 = sat0.vel[2]

    #Define variables
    c = ref.Sch_c
    n = ref.ang_vel
    k = ref.Sch_k
    h = 3*J2*R_e**2/(800*r_ref**2)
    #h = 6*J2*R_e**2/(r_ref**2)

    i_ref = ref.inc_0 -(3*n*J2*R_e**2)/(2*k*r_ref**2)*np.cos(ref.inc_0)*np.sin(ref.inc_0)#Inclination of the reference orbit (and chief)

    i_sat = dz_0/(k*r_ref)+i_ref

    if i_ref == 0:
        omega_0 = 0
    else:
        omega_0 = z_0/(r_ref*np.sin(i_ref))

    if (omega_0 and i_ref) != 0:
        gamma_0 = np.arctan(1/((1/np.tan(i_ref)*np.sin(i_sat)-np.cos(i_sat)*np.cos(omega_0))/np.sin(omega_0)))
    else:
        gamma_0 = 0

    phi_0 = np.arccos(np.cos(i_sat)*np.cos(i_ref)+np.sin(i_sat)*np.sin(i_ref)*np.cos(omega_0))
    d_omega_sat = -3*n*J2*R_e**2/(2*r_ref**2)*np.cos(i_sat)
    d_omega_ref = -3*n*J2*R_e**2/(2*r_ref**2)*np.cos(i_ref)

    temp = np.cos(gamma_0)*np.sin(gamma_0)*1/np.tan(omega_0)
    temp = temp if temp == temp else 0

    q = n*c - (temp-np.sin(gamma_0)**2*np.cos(i_sat))*(d_omega_sat - d_omega_ref)-d_omega_sat*np.cos(i_sat)
    l = -r_ref*np.sin(i_sat)*np.sin(i_ref)*np.sin(omega_0)/np.sin(phi_0)*(d_omega_sat-d_omega_ref)
    l = l if l == l else 0

    def equations(p):
        m,phi = p
        return(m*np.sin(phi)-z_0,l*np.sin(phi)+q*m*np.cos(phi)-dz_0)

    #Solve simultaneous equations
    m,phi = fsolve(equations,(0,0))


    #Equations of motion
    def J2_pert_func(t,state):
        [x,y,z] = state[:3] #Position
        [dx,dy,dz] = state[3:] #Velocity
        dX0 = dx
        dX1 = dy
        dX2 = dz

        theta = k*t

        gradJ2 = np.array([12*np.sin(i_ref)**2*np.cos(2*theta)*x + 8*np.sin(i_ref)**2*np.sin(2*theta)*y + 8*np.sin(2*i_ref)*np.sin(theta)*z,
                           8*np.sin(i_ref)**2*np.sin(2*theta)*x - 7*np.sin(i_ref)**2*np.cos(2*theta)*y - 2*np.sin(2*i_ref)*np.cos(theta)*z,
                           8*np.sin(2*i_ref)*np.sin(theta)*x - 2*np.sin(2*i_ref)*np.cos(theta)*y - 5*np.sin(i_ref)**2*np.cos(2*theta)*z])

        #gradJ2 = np.array([(1-3*np.sin(i_ref)**2*np.sin(theta)**2)*x + np.sin(i_ref)**2*np.sin(2*theta)*y + np.sin(2*i_ref)*np.sin(theta)*z,
        #          np.sin(i_ref)**2*np.sin(2*theta)*x + (-0.25 - np.sin(i_ref)**2*(0.5-7/4*np.sin(theta)**2))*y-0.25*np.sin(2*i_ref)*np.cos(theta)*z,
        #          np.sin(2*i_ref)*np.sin(theta)*x - 0.25*np.sin(2*i_ref)*np.cos(theta)*y + (-0.75+np.sin(i_ref)**2*(0.5 + 1.25*np.sin(theta)**2))*z])

        #gradJ2 -= np.array([4*ref.Sch_s/h*x,-ref.Sch_s/h*y,-3*ref.Sch_s/h*z])


        dX3 = 2*n*c*dy + (5*c**2-2)*n**2*x + h*n**2*gradJ2[0] - 3*n**2*J2*(R_e**2/r_ref)*(0.5 - ((3*np.sin(i_ref)**2*np.sin(k*t)**2)/2) - ((1+3*np.cos(2*i_ref))/8))
        dX4 = -2*n*c*dx + h*n**2*gradJ2[1] - 3*n**2*J2*(R_e**2/r_ref)*np.sin(i_ref)**2*np.sin(k*t)*np.cos(k*t)
        dX5 = -(3*c**2-2)*n**2*z + h*n**2*gradJ2[2]
        #dX5 = -q**2*z + 2*l*q*np.cos(q*t+phi)

        return np.array([dX0,dX1,dX2,dX3,dX4,dX5])

    return J2_pert_func

#Equations of motion
J2_func0 = J2_pert_Mike(chief_0.to_Curvy(ref_orbit=True),ref)
J2_func1 = J2_pert_Mike(deputy1_0.to_Curvy(ref_orbit=True),ref)
J2_func2 = J2_pert_Mike(deputy2_0.to_Curvy(ref_orbit=True),ref)

X2_d0 = solve_ivp(J2_func0, [times[0],times[-1]], chief_0.to_Curvy(ref_orbit=True).state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X2_d0.success:
    raise Exception("Integration failed!!!!")

X2_d1 = solve_ivp(J2_func1, [times[0],times[-1]], deputy1_0.to_Curvy(ref_orbit=True).state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X2_d1.success:
    raise Exception("Integration failed!!!!")

X2_d2 = solve_ivp(J2_func2, [times[0],times[-1]], deputy2_0.to_Curvy(ref_orbit=True).state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
if not X2_d2.success:
    raise Exception("Integration failed!!!!")

chief_p_states_mike = X2_d0.y.transpose()
deputy1_p_states_mike = X2_d1.y.transpose()# - chief_p_states_mike
deputy2_p_states_mike = X2_d2.y.transpose()# - chief_p_states_mike

print("Done Mike")
#------------------------------------------------------------------------------------------
### Old version (LVLH Matrix) ###

rtol = 1e-9
atol = 1e-12
step = 100

#Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
X3_c = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], chief_0.to_Curvy(ref_orbit=True).state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X3_c.success:
    raise Exception("Integration failed!!!!")

X3_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], deputy1_0.to_Curvy(ref_orbit=True).state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X3_d1.success:
    raise Exception("Integration failed!!!!")

X3_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], deputy2_0.to_Curvy(ref_orbit=True).state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
if not X3_d2.success:
    raise Exception("Integration failed!!!!")

chief_p_states_old = X3_c.y.transpose()
deputy1_p_states_old = X3_d1.y.transpose()# - chief_p_states_old
deputy2_p_states_old = X3_d2.y.transpose()# - chief_p_states_old

print("Done Old")
#------------------------------------------------------------------------------------------
### Numerical Schweighart ###
rtol = 1e-12
atol = 1e-18
step = 1

#Equations of motion
J2_func0 = J2_pert_num(chief_0.to_Curvy(ref_orbit=True),ref)
J2_func1 = J2_pert_num(deputy1_0.to_Curvy(ref_orbit=True),ref)
J2_func2 = J2_pert_num(deputy2_0.to_Curvy(ref_orbit=True),ref)

X2_d0 = solve_ivp(J2_func0, [times[0],times[-1]], chief_0.to_Curvy(ref_orbit=True).state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X2_d0.success:
    raise Exception("Integration failed!!!!")

X2_d1 = solve_ivp(J2_func1, [times[0],times[-1]], deputy1_0.to_Curvy(ref_orbit=True).state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X2_d1.success:
    raise Exception("Integration failed!!!!")

X2_d2 = solve_ivp(J2_func2, [times[0],times[-1]], deputy2_0.to_Curvy(ref_orbit=True).state, t_eval = times, rtol = rtol, atol = atol, max_step=step)
if not X2_d2.success:
    raise Exception("Integration failed!!!!")

chief_p_states_num = X2_d0.y.transpose()
deputy1_p_states_num = X2_d1.y.transpose()# - chief_p_states_num
deputy2_p_states_num = X2_d2.y.transpose()# - chief_p_states_num

print("Done Numerical")
#------------------------------------------------------------------------------------------
c_eci = []
d1_eci = []
d2_eci = []
c_sol = []
d1_sol = []
d2_sol = []
c_mike = []
d1_mike = []
d2_mike = []
c_num = []
d1_num = []
d2_num = []
c_old = []
d1_old = []
d2_old = []

print("Integration Done")
for i in range(len(times)):
    pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(times[i],True)
    c_eci.append(orbits.ECI_Sat(chief_p_states_eci[i,:3],chief_p_states_eci[i,3:],times[i],ref).to_LVLH(ref_orbit=True))
    d1_eci.append(orbits.ECI_Sat(deputy1_p_states_eci[i,:3],deputy1_p_states_eci[i,3:],times[i],ref).to_LVLH(ref_orbit=True))
    d2_eci.append(orbits.ECI_Sat(deputy2_p_states_eci[i,:3],deputy2_p_states_eci[i,3:],times[i],ref).to_LVLH(ref_orbit=True))

    c_sol.append(orbits.Curvy_Sat(chief_p_states_sol[i,:3],chief_p_states_sol[i,3:],times[i],ref).to_LVLH(ref_orbit=True))
    d1_sol.append(orbits.Curvy_Sat(deputy1_p_states_sol[i,:3],deputy1_p_states_sol[i,3:],times[i],ref).to_LVLH(ref_orbit=True))
    d2_sol.append(orbits.Curvy_Sat(deputy2_p_states_sol[i,:3],deputy2_p_states_sol[i,3:],times[i],ref).to_LVLH(ref_orbit=True))

    c_mike.append(orbits.Curvy_Sat(chief_p_states_mike[i,:3],chief_p_states_mike[i,3:],times[i],ref).to_LVLH(ref_orbit=True))
    d1_mike.append(orbits.Curvy_Sat(deputy1_p_states_mike[i,:3],deputy1_p_states_mike[i,3:],times[i],ref).to_LVLH(ref_orbit=True))
    d2_mike.append(orbits.Curvy_Sat(deputy2_p_states_mike[i,:3],deputy2_p_states_mike[i,3:],times[i],ref).to_LVLH(ref_orbit=True))

    c_num.append(orbits.Curvy_Sat(chief_p_states_num[i,:3],chief_p_states_num[i,3:],times[i],ref).to_LVLH(ref_orbit=True))
    d1_num.append(orbits.Curvy_Sat(deputy1_p_states_num[i,:3],deputy1_p_states_num[i,3:],times[i],ref).to_LVLH(ref_orbit=True))
    d2_num.append(orbits.Curvy_Sat(deputy2_p_states_num[i,:3],deputy2_p_states_num[i,3:],times[i],ref).to_LVLH(ref_orbit=True))

    c_old.append(orbits.Curvy_Sat(chief_p_states_old[i,:3],chief_p_states_old[i,3:],times[i],ref).to_LVLH(ref_orbit=True))
    d1_old.append(orbits.Curvy_Sat(deputy1_p_states_old[i,:3],deputy1_p_states_old[i,3:],times[i],ref).to_LVLH(ref_orbit=True))
    d2_old.append(orbits.Curvy_Sat(deputy2_p_states_old[i,:3],deputy2_p_states_old[i,3:],times[i],ref).to_LVLH(ref_orbit=True))
print("Classifying Done")

#--------------------------------------------------------------------------------------------- #
#Separations and accelerations
rel_d1_eci = np.zeros((n_times,3))
rel_d2_eci = np.zeros((n_times,3))
rel_d1_sol = np.zeros((n_times,3))
rel_d2_sol = np.zeros((n_times,3))
rel_d1_mike = np.zeros((n_times,3))
rel_d2_mike = np.zeros((n_times,3))
rel_d1_num = np.zeros((n_times,3))
rel_d2_num = np.zeros((n_times,3))
rel_d1_old = np.zeros((n_times,3))
rel_d2_old = np.zeros((n_times,3))

for ix in range(n_times):
    #Component of perturbed orbit in rho direction
    rel_d1_eci[ix] = d1_eci[ix].pos - c_eci[ix].pos
    rel_d2_eci[ix] = d2_eci[ix].pos - c_eci[ix].pos
    rel_d1_sol[ix] = d1_sol[ix].pos - c_sol[ix].pos
    rel_d2_sol[ix] = d2_sol[ix].pos - c_sol[ix].pos
    rel_d1_mike[ix] = d1_mike[ix].pos - c_mike[ix].pos
    rel_d2_mike[ix] = d2_mike[ix].pos - c_mike[ix].pos
    rel_d1_num[ix] = d1_num[ix].pos - c_num[ix].pos
    rel_d2_num[ix] = d2_num[ix].pos - c_num[ix].pos
    rel_d1_old[ix] = d1_old[ix].pos - c_old[ix].pos
    rel_d2_old[ix] = d2_old[ix].pos - c_old[ix].pos

    rel_d1_eci[ix] = c_eci[ix].pos
    rel_d2_eci[ix] = c_eci[ix].pos
    rel_d1_sol[ix] = c_sol[ix].pos
    rel_d2_sol[ix] = c_sol[ix].pos
    rel_d1_mike[ix] = c_mike[ix].pos
    rel_d2_mike[ix] = c_mike[ix].pos
    rel_d1_num[ix] = c_num[ix].pos
    rel_d2_num[ix] = c_num[ix].pos
    rel_d1_old[ix] = c_old[ix].pos
    rel_d2_old[ix] = c_old[ix].pos

rel_d1 = np.array([rel_d1_eci,rel_d1_sol,rel_d1_num,rel_d1_mike,rel_d1_old])
rel_d2 = np.array([rel_d2_eci,rel_d2_sol,rel_d2_num,rel_d2_mike,rel_d2_old])
label = ["ECI", "Schweighart solved", "Schweighart Numerical", "Mike Schweighart", "Old Integration"]
# ---------------------------------------------------------------------- #

#Plot rho
axis = 0
plt.figure(1)
plt.clf()
for ix in range(3):
    for iy in range(3-ix):
        plt.subplot(3,3,6-ix*3+iy+1)
        plt.plot(times,rel_d1[ix,:,axis],'b-', label="Deputy 1, Y-axis")
        plt.plot(times,rel_d2[ix,:,axis],'r-', label="Deputy 2, Y-axis")
        plt.plot(times,rel_d1[3-iy,:,axis],'b--', label="Deputy 1, X-axis")
        plt.plot(times,rel_d2[3-iy,:,axis],'r--', label="Deputy 2, X-axis")
        plt.plot(times,rel_d1[ix,:,axis] - rel_d1[3-iy,:,axis],'c--', label="Deputy 1 Residuals")
        plt.plot(times,rel_d2[ix,:,axis] - rel_d2[3-iy,:,axis],'m--', label="Deputy 2 Residuals")
        if ix == 0:
            plt.xlabel(label[3-iy])
        if iy == 0:
            plt.ylabel(label[ix])
plt.suptitle('Differential Rho Separations against time')
plt.legend(loc=(2.4,-1))

#Plot xi
axis = 1
plt.figure(2)
plt.clf()
for ix in range(3):
    for iy in range(3-ix):
        plt.subplot(3,3,6-ix*3+iy+1)
        plt.plot(times,rel_d1[ix,:,axis],'b-', label="Deputy 1, Y-axis")
        plt.plot(times,rel_d2[ix,:,axis],'r-', label="Deputy 2, Y-axis")
        plt.plot(times,rel_d1[3-iy,:,axis],'b--', label="Deputy 1, X-axis")
        plt.plot(times,rel_d2[3-iy,:,axis],'r--', label="Deputy 2, X-axis")
        plt.plot(times,rel_d1[ix,:,axis] - rel_d1[3-iy,:,axis],'c--', label="Deputy 1 Residuals")
        plt.plot(times,rel_d2[ix,:,axis] - rel_d2[3-iy,:,axis],'m--', label="Deputy 2 Residuals")
        if ix == 0:
            plt.xlabel(label[3-iy])
        if iy == 0:
            plt.ylabel(label[ix])
plt.suptitle('Differential Xi Separations against time')
plt.legend(loc=(2.4,-1))


#Plot eta
axis = 2
plt.figure(3)
plt.clf()
for ix in range(3):
    for iy in range(3-ix):
        plt.subplot(3,3,6-ix*3+iy+1)
        plt.plot(times,rel_d1[ix,:,axis],'b-', label="Deputy 1, Y-axis")
        plt.plot(times,rel_d2[ix,:,axis],'r-', label="Deputy 2, Y-axis")
        plt.plot(times,rel_d1[3-iy,:,axis],'b--', label="Deputy 1, X-axis")
        plt.plot(times,rel_d2[3-iy,:,axis],'r--', label="Deputy 2, X-axis")
        plt.plot(times,rel_d1[ix,:,axis] - rel_d1[3-iy,:,axis],'c--', label="Deputy 1 Residuals")
        plt.plot(times,rel_d2[ix,:,axis] - rel_d2[3-iy,:,axis],'m--', label="Deputy 2 Residuals")
        if ix == 0:
            plt.xlabel(label[3-iy])
        if iy == 0:
            plt.ylabel(label[ix])
plt.suptitle('Differential Eta Separations against time')
plt.legend(loc=(2.4,-1))

#Plot rho residuals
axis = 0
plt.figure(4)
plt.clf()
for ix in range(3):
    for iy in range(3-ix):
        plt.subplot(3,3,6-ix*3+iy+1)
        plt.plot(times,rel_d1[ix,:,axis] - rel_d1[3-iy,:,axis],'c--', label="Deputy 1 Residuals")
        plt.plot(times,rel_d2[ix,:,axis] - rel_d2[3-iy,:,axis],'m--', label="Deputy 2 Residuals")
        if ix == 0:
            plt.xlabel(label[3-iy])
        if iy == 0:
            plt.ylabel(label[ix])
plt.suptitle('Residual Differential Rho Separations against time')
plt.legend(loc=(2.4,-1))

#Plot xi residuals
axis = 1
plt.figure(5)
plt.clf()
for ix in range(3):
    for iy in range(3-ix):
        plt.subplot(3,3,6-ix*3+iy+1)
        plt.plot(times,rel_d1[ix,:,axis] - rel_d1[3-iy,:,axis],'c--', label="Deputy 1 Residuals")
        plt.plot(times,rel_d2[ix,:,axis] - rel_d2[3-iy,:,axis],'m--', label="Deputy 2 Residuals")
        if ix == 0:
            plt.xlabel(label[3-iy])
        if iy == 0:
            plt.ylabel(label[ix])
plt.suptitle('Residual Differential Xi Separations against time')
plt.legend(loc=(2.4,-1))

#Plot eta residuals
axis = 2
plt.figure(6)
plt.clf()
for ix in range(3):
    for iy in range(3-ix):
        plt.subplot(3,3,6-ix*3+iy+1)
        plt.plot(times,rel_d1[ix,:,axis] - rel_d1[3-iy,:,axis],'c--', label="Deputy 1 Residuals")
        plt.plot(times,rel_d2[ix,:,axis] - rel_d2[3-iy,:,axis],'m--', label="Deputy 2 Residuals")
        if ix == 0:
            plt.xlabel(label[3-iy])
        if iy == 0:
            plt.ylabel(label[ix])
plt.suptitle('Residual Differential Eta Separations against time')
plt.legend(loc=(2.4,-1))
