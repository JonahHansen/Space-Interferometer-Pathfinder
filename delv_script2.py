from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits as orbits
import sys
import copy
from scipy.optimize import minimize
from matplotlib.collections import LineCollection

plt.ion()

def cost_function(sig_chief, sig_state1, sig_state2, delv):
    kappa_c = 0.0001*np.array([1,1,1,1,1,1])
    kappa_d1 = np.array([1000,1,1,1000,1,1])
    kappa_d2 = np.array([1000,1,1,1000,1,1])
    kappa_dv = 1
    phi = np.dot(kappa_c,sig_chief**2) + np.dot(kappa_d1,sig_state1**2) + np.dot(kappa_d1,sig_state1**2) + kappa_dv*delv**2
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

def correct_orbit(ref,c0,d10,d20,t0,tfinal,t_burn1,t_burn2):
    #Tolerance and steps required for the integrator
    rtol = 1e-12
    atol = 1e-18

    t1 = np.linspace(t0,t_burn1,50)
    t2 = np.linspace(t_burn1,t_burn2,50)
    t3 = np.linspace(t_burn2,tfinal,50)

    times = np.array([t1,t2,t3])

    def optimiser(dvls):
        #import pdb; pdb.set_trace()
        chief_states = np.zeros((50*3,6))
        deputy1_states = np.zeros((50*3,6))
        deputy2_states = np.zeros((50*3,6))
        delv_bank2 = np.zeros((2,3,3))

        dvls = dvls.reshape((2,3,3))
        #print(dvls)
        #print(c0)

        c = c0
        d1 = d10
        d2 = d20

        for i in range(2):
            #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
            X_c = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[i,0],times[i,-1]], c, t_eval = times[i], rtol = rtol, atol = atol)
            #Check if successful integration
            if not X_c.success:
                raise Exception("Integration Chief failed!!!!")

            #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
            X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[i,0],times[i,-1]], d1, t_eval = times[i], rtol = rtol, atol = atol)
            #Check if successful integration
            if not X_d1.success:
                raise Exception("Integration Deputy 1 failed!!!!")

            X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[i,0],times[i,-1]], d2, t_eval = times[i], rtol = rtol, atol = atol)
            if not X_d2.success:
                raise Exception("Integration Deputy 2 failed!!!!")

            chief_p_states = X_c.y.transpose()
            deputy1_p_states = X_d1.y.transpose()
            deputy2_p_states = X_d2.y.transpose()

            chief_states[i*50:i*50+50] = chief_p_states
            deputy1_states[i*50:i*50+50] = deputy1_p_states
            deputy2_states[i*50:i*50+50] = deputy2_p_states

            #import pdb; pdb.set_trace()

            delv_c = dvls[i,0]
            delv_d1 = dvls[i,1]
            delv_d2 = dvls[i,2]

            delv_bank2[i] = np.array([delv_c,delv_d1,delv_d2])

            c = chief_states[50+i*50-1] + np.append(np.zeros(3),delv_c)
            d1 = deputy1_states[50+i*50-1] + np.append(np.zeros(3),delv_d1)
            d2 = deputy2_states[50+i*50-1] + np.append(np.zeros(3),delv_d2)

        #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
        X_c = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[2,0],times[2,-1]], c, t_eval = times[2], rtol = rtol, atol = atol)
        #Check if successful integration
        if not X_c.success:
            raise Exception("Integration Chief failed!!!!")

        #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
        X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[2,0],times[2,-1]], d1, t_eval = times[2], rtol = rtol, atol = atol)
        #Check if successful integration
        if not X_d1.success:
            raise Exception("Integration Deputy 1 failed!!!!")

        X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[2,0],times[2,-1]], d2, t_eval = times[2], rtol = rtol, atol = atol)
        if not X_d2.success:
            raise Exception("Integration Deputy 2 failed!!!!")

        chief_p_states = X_c.y.transpose()
        deputy1_p_states = X_d1.y.transpose()
        deputy2_p_states = X_d2.y.transpose()

        chief_states[100:150] = chief_p_states
        deputy1_states[100:150] = deputy1_p_states
        deputy2_states[100:150] = deputy2_p_states

        c_final = state=chief_states[-1]
        d1_final = orbits.ECI_Sat(deputy1_p_states[-1,:3],deputy1_p_states[-1,3:],times[2,-1],ref).to_Baseline(state=chief_p_states[-1])
        d2_final = orbits.ECI_Sat(deputy2_p_states[-1,:3],deputy2_p_states[-1,3:],times[2,-1],ref).to_Baseline(state=chief_p_states[-1])

        c_true = orbits.init_chief(ref,time=times[2,-1]).state
        d1_true = orbits.init_deputy(ref,1,time=times[2,-1]).to_Baseline(state=c_true)
        d2_true = orbits.init_deputy(ref,2,time=times[2,-1]).to_Baseline(state=c_true)

        print(c_final-c_true)
        print(d1_final.state-d1_true.state)
        print(d2_final.state-d2_true.state)
        print(np.sum(delv_bank2))
        PHI = cost_function(c_final-c_true, d1_final.state-d1_true.state, d2_final.state - d2_true.state, np.sum(delv_bank2))
        print(PHI)
        #import pdb; pdb.set_trace()
        return PHI

    delvs = np.zeros((2,3,3))

    x = minimize(optimiser,delvs)

    return x

##################################################
plt.ion()

alt = 500e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(40) #20
#Longitude of the Ascending Node
Om_0 = np.radians(90) #0

#Stellar vector
ra = np.radians(29) #90
dec = np.radians(45)#-40

#The max distance to the other satellites in m
delta_r_max = 0.3e3

#Tolerance and steps required for the integrator
rtol = 1e-12
atol = 1e-18

zeta = float(sys.argv[2])

def del_v_func(c,d1,d2,t,pt,ref):

    sat0 = c.to_Baseline(state=c.state)
    sat1 = d1.to_Baseline(state=c.state)
    sat2 = d2.to_Baseline(state=c.state)

    csat = sat0.state
    dsat1 = sat1.state
    dsat2 = sat2.state

    delvs1 = np.zeros(3)
    delvs2 = np.zeros(3)
    delvs0 = np.zeros(3)

    max_s_sep = np.max([csat[2],dsat1[2],dsat2[2]])
    #print(max_s_sep)

    del_t = (t-pt)/zeta

    delvs0[2] = (max_s_sep - csat[2])/del_t
    delvs1[2] = (max_s_sep - dsat1[2])/del_t
    delvs2[2] = (max_s_sep - dsat2[2])/del_t

    b1 = np.linalg.norm(dsat1[0:2])
    b2 = np.linalg.norm(dsat2[0:2])
    del_b = b2-b1
    #print(del_b)

    if del_b >= 0:
        delvs2[2] += del_b/del_t
    elif del_b < 0:
        delvs1[2] += -del_b/del_t

    delv = np.array([delvs0[2],delvs1[2],delvs2[2]])
    #print(delv)
    sat0.vel += delvs0
    sat1.vel += delvs1
    sat2.vel += delvs2

    new_sat0 = sat0.to_ECI(state = c.state).state
    new_sat1 = sat1.to_ECI(state = c.state).state
    new_sat2 = sat2.to_ECI(state = c.state).state

    return delv,new_sat0,new_sat1,new_sat2


#------------------------------------------------------------------------------------------
#Calculate reference orbit, in the geocentric (ECI) frame (See Orbit module)
ref = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

#Number of orbits
n_orbits = 0.5
#Number of phases in each orbit
n_phases = 500
#Total evaluation points
n_times = int(n_orbits*n_phases)
times = np.linspace(0,ref.period*n_orbits,n_times) #Create list of times

#Initial states of the satellites
chief_0 = orbits.init_chief(ref).state
deputy1_0 = orbits.init_deputy(ref,1).state
deputy2_0 = orbits.init_deputy(ref,2).state

chief_states = np.array([chief_0])
deputy1_states = np.array([deputy1_0])
deputy2_states = np.array([deputy2_0])
delv_bank = []

t_burn = int(sys.argv[1])
t0 = 0.0
t_bank = np.array([0])

while t0 < times[-1]:
    burn_pt = t0 + t_burn
    ts = np.linspace(t0,burn_pt,10) #Every 0.1s
    t_bank = np.append(t_bank,ts)

    #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
    X_c = solve_ivp(lambda t, y: dX_dt(t,y,ref), [ts[0],ts[-1]], chief_0, t_eval = ts, rtol = rtol, atol = atol)
    #Check if successful integration
    if not X_c.success:
        raise Exception("Integration Chief failed!!!!")

    #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
    X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [ts[0],ts[-1]], deputy1_0, t_eval = ts, rtol = rtol, atol = atol)
    #Check if successful integration
    if not X_d1.success:
        raise Exception("Integration Deputy 1 failed!!!!")

    X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [ts[0],ts[-1]], deputy2_0, t_eval = ts, rtol = rtol, atol = atol)
    if not X_d2.success:
        raise Exception("Integration Deputy 2 failed!!!!")

    chief_p_states = X_c.y.transpose()
    deputy1_p_states = X_d1.y.transpose()
    deputy2_p_states = X_d2.y.transpose()

    chief_states = np.append(chief_states,chief_p_states,axis=0)
    deputy1_states = np.append(deputy1_states,deputy1_p_states,axis=0)
    deputy2_states = np.append(deputy2_states,deputy2_p_states,axis=0)

    c = orbits.ECI_Sat(chief_p_states[-1,:3],chief_p_states[-1,3:],ts[-1],ref)
    d1 = orbits.ECI_Sat(deputy1_p_states[-1,:3],deputy1_p_states[-1,3:],ts[-1],ref)
    d2 = orbits.ECI_Sat(deputy2_p_states[-1,:3],deputy2_p_states[-1,3:],ts[-1],ref)

    delv,new_c,new_d1,new_d2 = del_v_func(c,d1,d2,burn_pt,t0,ref)
    #import pdb; pdb.set_trace()
    delv_bank.append(delv)
    chief_0 = new_c
    deputy1_0 = new_d1
    deputy2_0 = new_d2
    t0 = burn_pt
    #print(t0)

c0 = chief_states[-1]
d10 = deputy1_states[-1]
d20 = deputy2_states[-1]
t0 = times[-1]
t_end = ref.period
t_burn1 = t0 + 60*5
t_burn2 = t_end - 60*10

y = correct_orbit(ref,c0,d10,d20,t0,t_end,t_burn1,t_burn2)
