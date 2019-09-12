import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits as orbits
import copy
from scipy.optimize import minimize

#plt.ion()

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

def cost_function(sig_state1, sig_state2, d1, d2, delv):
    kappa_d1 = 10*np.array([10,1,1e7,10,10,1e9])
    kappa_d2 = 10*np.array([10,1,1e7,10,10,1e9])

    br = np.linalg.norm(d1[:3]) - np.linalg.norm(d2[:3])
    bv = np.linalg.norm(d1[3:]) - np.linalg.norm(d2[3:])

    kappa_br = 1e7
    kappa_bv = 1e10
    #kappa_b = 5000*np.array([1,1,1,1,1,1])
    kappa_dv = np.array([1,1,1])
    phi = np.dot(kappa_d1,sig_state1**2) + np.dot(kappa_d1,sig_state2**2) + np.dot(kappa_dv,delv**2) + kappa_br*br**2 + kappa_br*bv**2#+ np.dot(kappa_b,baseline**2)
    return phi

def del_v_func(c,d1,d2,t,pt,ref,zeta):

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

def integration_fix(ref, c0, d10, d20, t0, t_final, t_burn, zeta):
    #Tolerance and steps required for the integrator
    rtol = 1e-12
    atol = 1e-18

    chief_states = np.array([c0])
    deputy1_states = np.array([d10])
    deputy2_states = np.array([d20])
    delv_bank = []

    t_bank = np.array([t0])

    while t0 < t_final:
        burn_pt = t0 + t_burn
        ts = np.linspace(t0,burn_pt,10) #Every 1s
        t_bank = np.append(t_bank,ts)

        #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
        X_c = solve_ivp(lambda t, y: dX_dt(t,y,ref), [ts[0],ts[-1]], c0, t_eval = ts, rtol = rtol, atol = atol)
        #Check if successful integration
        if not X_c.success:
            raise Exception("Integration Chief failed!!!!")

        #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
        X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [ts[0],ts[-1]], d10, t_eval = ts, rtol = rtol, atol = atol)
        #Check if successful integration
        if not X_d1.success:
            raise Exception("Integration Deputy 1 failed!!!!")

        X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [ts[0],ts[-1]], d20, t_eval = ts, rtol = rtol, atol = atol)
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

        delv,new_c,new_d1,new_d2 = del_v_func(c,d1,d2,burn_pt,t0,ref,zeta)

        delv_bank.append(delv)
        c0 = new_c
        d10 = new_d1
        d20 = new_d2
        t0 = burn_pt

    return chief_states,deputy1_states,deputy2_states,delv_bank,t_bank

def plotit(num,ref,t_bank,chief_states,deputy1_states,deputy2_states):

    rel_p_dep1 = []
    rel_p_dep2 = []

    for i in range(len(t_bank)):
        rel_p_dep1.append(orbits.ECI_Sat(deputy1_states[i,:3],deputy1_states[i,3:],t_bank[i],ref).to_Baseline(state=chief_states[i]))
        rel_p_dep2.append(orbits.ECI_Sat(deputy2_states[i,:3],deputy2_states[i,3:],t_bank[i],ref).to_Baseline(state=chief_states[i]))

    n_times = len(t_bank)
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
        baseline_sep[ix] = np.linalg.norm(rel_p_dep2[ix].pos) - np.linalg.norm(rel_p_dep1[ix].pos)

        #Component of perturbed orbit in star direction
        s_hat_drd1[ix] = rel_p_dep1[ix].pos[2]
        s_hat_drd2[ix] = rel_p_dep2[ix].pos[2]

        #Component of perturbed orbit in baseline direction
        b_hat_drd1[ix] = rel_p_dep1[ix].pos[0]
        b_hat_drd2[ix] = rel_p_dep2[ix].pos[0]

        #Separation of the two deputies in the star direction
        s_hat_sep[ix] = s_hat_drd2[ix] - s_hat_drd1[ix]
        #baseline_sep[ix] = b_hat_drd1[ix] + b_hat_drd2[ix]
        #Delta of the separation along the star direction and the baseline direction
        total_sep[ix] = baseline_sep[ix] - s_hat_sep[ix]

    print("Total sep = " + str(np.max(np.abs(total_sep))))

    # ---------------------------------------------------------------------- #
    ### PLOTTING STUFF ###

    #Plot separation along the star direction
    plt.figure(num*3+1)
    plt.clf()
    plt.plot(t_bank,s_hat_drd1,"b-",label="Deputy 1, s direction")
    plt.plot(t_bank,s_hat_drd2,"g-",label="Deputy 2, s direction")
    plt.plot(t_bank,s_hat_sep,"r-",label="Separation, s direction")
    plt.xlabel("Times(s)")
    plt.ylabel("Separation(m)")
    plt.title('Separations against time due to perturbations')
    plt.legend()

    #Plot separation along the baseline direction
    plt.figure(num*3+2)
    plt.clf()
    plt.plot(t_bank,baseline_sep,"y-",label="Separation, baseline direction")
    #plt.plot(times,total_sep,"c-",label="Total direction")
    plt.xlabel("Times(s)")
    plt.ylabel("Separation(m)")
    plt.title('Separations against time due to perturbations')
    plt.legend()

    #Plot separation in the baseline frame
    plt.figure(num*3+3)
    plt.clf()
    plt.plot(t_bank,total_sep,"c-",label="Total direction")
    plt.xlabel("Times(s)")
    plt.ylabel("Separation(m)")
    plt.title('Separations against time due to perturbations')
    plt.legend()

    return

def recharge_fix(ref,c0,d10,d20,n_burns,burn_times):
    #Tolerance and steps required for the integrator
    rtol = 1e-12
    atol = 1e-18
    times = np.zeros((n_burns+1,50))

    for i in range(n_burns+1):
        times[i] = np.linspace(burn_times[i],burn_times[i+1],50)

    def correct_orbit(delvs):
        #import pdb; pdb.set_trace()
        chief_states = np.zeros((50*(n_burns+1),6))
        deputy1_states = np.zeros((50*(n_burns+1),6))
        deputy2_states = np.zeros((50*(n_burns+1),6))
        delv_bank2 = np.zeros((n_burns,3))

        delvs = delvs.reshape((n_burns,3,3))
        #print(dvls)
        #print(c0)

        c = c0
        d1 = d10
        d2 = d20

        for i in range(n_burns):
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

            delv_c = delvs[i,0]
            delv_d1 = delvs[i,1]
            delv_d2 = delvs[i,2]

            delv_bank2[i] = np.array([np.linalg.norm(delv_c),np.linalg.norm(delv_d1),np.linalg.norm(delv_d2)])

            c = chief_states[50+i*50-1] + np.append(np.zeros(3),delv_c)
            d1 = deputy1_states[50+i*50-1] + np.append(np.zeros(3),delv_d1)
            d2 = deputy2_states[50+i*50-1] + np.append(np.zeros(3),delv_d2)

        #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
        X_c = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[n_burns,0],times[n_burns,-1]], c, t_eval = times[n_burns], rtol = rtol, atol = atol)
        #Check if successful integration
        if not X_c.success:
            raise Exception("Integration Chief failed!!!!")

        #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
        X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[n_burns,0],times[n_burns,-1]], d1, t_eval = times[n_burns], rtol = rtol, atol = atol)
        #Check if successful integration
        if not X_d1.success:
            raise Exception("Integration Deputy 1 failed!!!!")

        X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[n_burns,0],times[n_burns,-1]], d2, t_eval = times[n_burns], rtol = rtol, atol = atol)
        if not X_d2.success:
            raise Exception("Integration Deputy 2 failed!!!!")

        chief_p_states = X_c.y.transpose()
        deputy1_p_states = X_d1.y.transpose()
        deputy2_p_states = X_d2.y.transpose()

        chief_states[n_burns*50:n_burns*50+50] = chief_p_states
        deputy1_states[n_burns*50:n_burns*50+50] = deputy1_p_states
        deputy2_states[n_burns*50:n_burns*50+50] = deputy2_p_states

        return chief_states, deputy1_states, deputy2_states, delv_bank2

    def optimiser(delvs):
        chief_states, deputy1_states, deputy2_states, delv_bank = correct_orbit(delvs)

        c_final = state=chief_states[-1]
        d1_final = orbits.ECI_Sat(deputy1_states[-1,:3],deputy1_states[-1,3:],times[n_burns,-1],ref).to_Baseline(state=c_final).state
        d2_final = orbits.ECI_Sat(deputy2_states[-1,:3],deputy2_states[-1,3:],times[n_burns,-1],ref).to_Baseline(state=c_final).state

        d1_true = orbits.init_deputy(ref,1,time=times[n_burns,-1],ref_orbit=False,state=c_final).to_Baseline(state=c_final).state
        d2_true = orbits.init_deputy(ref,2,time=times[n_burns,-1],ref_orbit=False,state=c_final).to_Baseline(state=c_final).state

        #print(d1_final.state-d1_true.state)
        #print(d2_final.state-d2_true.state)
        #print(np.sum(delv_bank,axis=0))
        PHI = cost_function(d1_final-d1_true, d2_final - d2_true, d1_final, d2_final, np.sum(delv_bank,axis=0))
        print(PHI)
        #import pdb; pdb.set_trace()
        return PHI

    delvs = np.zeros((n_burns,3,3))
    x = minimize(optimiser,delvs,method="Nelder-Mead")
    chief_states, deputy1_states, deputy2_states, delv_bank = correct_orbit(x.x)

    return chief_states, deputy1_states, deputy2_states, delv_bank, times.reshape(np.size(times))
