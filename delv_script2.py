from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits as orbits
import sys
from matplotlib.collections import LineCollection

plt.ion()

def cost_function(sig_state1, sig_state2, delv):
    kappa_d1 = np.array([1,2,3,4,5,6])
    kappa_d2 = np.array([1,2,3,4,5,6])
    kappa_dv =
    phi = np.dot(kappa_d1,sig_state1**2) + np.dot(kappa_d1,sig_state1**2) + kappa_dv*delv
    return phi

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

def correct_orbit(c0,d10,d20,t0,tfinal,t_burn1,t_burn2):
    #Tolerance and steps required for the integrator
    rtol = 1e-12
    atol = 1e-18

    chief_states = np.zeros((50*3,6))
    deputy1_states = np.zeros((50*3,6))
    deputy2_states = np.zeros((50*3,6))
    delv_bank = np.zeros((2,3,3))

    t1 = np.linspace(t0,t_burn1,50)
    t2 = np.linspace(t_burn1,t_burn2,50)
    t3 = np/linspace(t_burn2,tfinal,50)

    times = np.array([t1,t2,t3])

    def optimiser(dvls):
        for i in range(2):
            #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
            X_c = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[i,0],times[i,-1]], c0, t_eval = times[i], rtol = rtol, atol = atol)
            #Check if successful integration
            if not X_c.success:
                raise Exception("Integration Chief failed!!!!")

            #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
            X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[i,0],times[i,-1]], d10, t_eval = times[i], rtol = rtol, atol = atol)
            #Check if successful integration
            if not X_d1.success:
                raise Exception("Integration Deputy 1 failed!!!!")

            X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[i,0],times[i,-1]], d20, t_eval = times[i], rtol = rtol, atol = atol)
            if not X_d2.success:
                raise Exception("Integration Deputy 2 failed!!!!")

            chief_p_states = X_c.y.transpose()
            deputy1_p_states = X_d1.y.transpose()
            deputy2_p_states = X_d2.y.transpose()

            chief_states[i*50:i*50+50] = chief_p_states
            deputy1_states[i*50:i*50+50] = deputy1_p_states
            deputy2_states[i*50:i*50+50] = deputy2_p_states

            delv_c = dvls[i,0]
            delv_d1 = dvls[i,1]
            delv_d2 = dvls[i,2]

            delv_bank[i] = np.array([delv_c,delv_d1,delv_d2]))

            c0 += np.append(np.zeros(3),delv_c)
            d10 += np.append(np.zeros(3),delv_d1)
            d20 += np.append(np.zeros(3),delv_d2)

        #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
        X_c = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[i,0],times[i,-1]], c0, t_eval = times[i], rtol = rtol, atol = atol)
        #Check if successful integration
        if not X_c.success:
            raise Exception("Integration Chief failed!!!!")

        #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
        X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[i,0],times[i,-1]], d10, t_eval = times[i], rtol = rtol, atol = atol)
        #Check if successful integration
        if not X_d1.success:
            raise Exception("Integration Deputy 1 failed!!!!")

        X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[i,0],times[i,-1]], d20, t_eval = times[i], rtol = rtol, atol = atol)
        if not X_d2.success:
            raise Exception("Integration Deputy 2 failed!!!!")

        chief_p_states = X_c.y.transpose()
        deputy1_p_states = X_d1.y.transpose()
        deputy2_p_states = X_d2.y.transpose()

        chief_states[100:150] = chief_p_states
        deputy1_states[100:150] = deputy1_p_states
        deputy2_states[100:150] = deputy2_p_states

        c_final = orbits.ECI_Sat(chief_p_states[-1,:3],chief_p_states[-1,3:],ts[-1],ref).to_Baseline
        d1_final = orbits.ECI_Sat(deputy1_p_states[-1,:3],deputy1_p_states[-1,3:],ts[-1],ref).to_Baseline
        d2_final = orbits.ECI_Sat(deputy2_p_states[-1,:3],deputy2_p_states[-1,3:],ts[-1],ref).to_Baseline

        return cost_function()
