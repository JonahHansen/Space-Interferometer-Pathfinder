import numpy as np
import astropy.constants as const
import modules.orbits as orbits
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

#Set up orbital configuration

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

#Calculate reference orbit, in the geocentric (ECI) frame (See Orbit module)
ref = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)


""" J2 Perturbation acceleration function
    Takes in current time, state and reference orbit
    Returns the acceleration """
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

""" Propagates the orbit using the ECI J2 integrator
    Takes in the initial state, a list of times and the reference orbits
    Returns the states at the given times """
def propagate_orbit(X0, times, ref):
    rtol = 1e-12
    atol = 1e-18

    #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
    X = solve_ivp(lambda t, y: dX_dt(t,y,ref), [times[0],times[-1]], X0, t_eval = times, rtol = rtol, atol = atol)
    #Check if successful integration
    if not X.success:
        raise Exception("Integration failed!!!!")

    states = X.y.transpose()
    return states

""" Calculate the max delta v required during the integration times
    Takes in the ref orbit, the chief and deputy states and the list of times
    Returns nothing, but prints the delta v to terminal """
def calc_delv(ref,c_states,d1_states,d2_states,times):
    dep1_base = []
    dep2_base = []

    #Convert states to baseline frame
    for i in range(len(times)):
        dep1_base.append(orbits.ECI_Sat(d1_states[i,:3],d1_states[i,3:],times[i],ref).to_Baseline(state=c_states[i]))
        dep2_base.append(orbits.ECI_Sat(d2_states[i,:3],d2_states[i,3:],times[i],ref).to_Baseline(state=c_states[i]))

    n_times = len(times)

    #Separations and accelerations
    baseline_sep = np.zeros(n_times) #Separation along the baseline
    s_hat_drd1 = np.zeros(n_times) #Deputy1 position in star direction
    s_hat_drd2 = np.zeros(n_times) #Deputy2 position in star direction

    for ix in range(n_times):
        #Baseline separations is simply the difference between the positions of the two deputies
        baseline_sep[ix] = np.linalg.norm(dep2_base[ix].pos) - np.linalg.norm(dep1_base[ix].pos)

        #Component of perturbed orbit in star direction
        s_hat_drd1[ix] = dep1_base[ix].pos[2]
        s_hat_drd2[ix] = dep2_base[ix].pos[2]

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

    #Delta v (Integral of the absolute value of the acceleration)
    delta_v_s1 = np.trapz(acc_s1,times)
    delta_v_s2 = np.trapz(acc_s2,times)
    delta_v_delta_b = np.trapz(acc_delta_b,times)

    print("Delta_v for integration is: " + str(delta_v_s1 + delta_v_s2 + delta_v_delta_b))
    return


def propagate_integration(ref, c0, d10, d20, t0, t_final):
    #Tolerance and steps required for the integrator
    times = np.linspace(t0,t_final,500)

    c_states = propagate_orbit(c0,times,ref)
    d1_states = propagate_orbit(d10,times,ref)
    d2_states = propagate_orbit(d20,times,ref)

    calc_delv(ref,c_states,d1_states,d2_states,times)

    d1sat = orbits.ECI_Sat(d1_states[-1,:3],d1_states[-1,3:],times[-1],ref).to_Baseline(state=c_states[-1])
    d2sat = orbits.ECI_Sat(d2_states[-1,:3],d2_states[-1,3:],times[-1],ref).to_Baseline(state=c_states[-1])

    d1sat.pos[2] = 0
    d2sat.pos[2] = 0

    d1_states[-1] = d1sat.to_ECI(state=c_states[-1]).state
    d2_states[-1] = d2sat.to_ECI(state=c_states[-1]).state

    return c_states,d1_states,d2_states,times


def cost_function(sig_state1, sig_state2, d1, d2, delv):
    kappa_d1 = 1000*np.array([1,1,3,2,2,3])
    kappa_d2 = 1000*np.array([1,1,3,2,2,3])

    br = np.linalg.norm(d1[:3]) - np.linalg.norm(d2[:3])
    bv = np.linalg.norm(d1[3:]) - np.linalg.norm(d2[3:])

    kappa_br = 0e8
    kappa_bv = 0e12
    #kappa_b = 5000*np.array([1,1,1,1,1,1])
    kappa_dv = np.array([1,1,1])
    phi = np.dot(kappa_d1,sig_state1**2) + np.dot(kappa_d1,sig_state2**2) + np.dot(kappa_dv,delv**2) + kappa_br*br**2 + kappa_br*bv**2#+ np.dot(kappa_b,baseline**2)
    return phi


def recharge_fix(ref,c0,d10,d20,n_burns,burn_times):
    times = np.zeros((n_burns+1,50))

    for i in range(n_burns+1):
        times[i] = np.linspace(burn_times[i],burn_times[i+1],50)

    def correct_orbit(delvs):
        c_states = np.zeros((50*(n_burns+1),6))
        d1_states = np.zeros((50*(n_burns+1),6))
        d2_states = np.zeros((50*(n_burns+1),6))
        delv_bank = np.zeros((n_burns,3))

        delvs = delvs.reshape((n_burns,2,3))

        c = c0
        d1 = d10
        d2 = d20

        for i in range(n_burns):
            c_states_part = propagate_orbit(c,times[i],ref)
            d1_states_part = propagate_orbit(d1,times[i],ref)
            d2_states_part = propagate_orbit(d2,times[i],ref)

            c_states[i*50:i*50+50] = c_states_part
            d1_states[i*50:i*50+50] = d1_states_part
            d2_states[i*50:i*50+50] = d2_states_part

            delv_c = 0
            delv_d1 = delvs[i,0]
            delv_d2 = delvs[i,1]

            delv_bank[i] = np.array([np.linalg.norm(delv_c),np.linalg.norm(delv_d1),np.linalg.norm(delv_d2)])

            c = c_states[50+i*50-1]# + np.append(np.zeros(3),delv_c)
            d1 = d1_states[50+i*50-1] + np.append(np.zeros(3),delv_d1)
            d2 = d2_states[50+i*50-1] + np.append(np.zeros(3),delv_d2)

        c_states_part = propagate_orbit(c,times[n_burns],ref)
        d1_states_part = propagate_orbit(d1,times[n_burns],ref)
        d2_states_part = propagate_orbit(d2,times[n_burns],ref)

        c_states[n_burns*50:n_burns*50+50] = c_states_part
        d1_states[n_burns*50:n_burns*50+50] = d1_states_part
        d2_states[n_burns*50:n_burns*50+50] = d2_states_part

        return c_states, d1_states, d2_states, delv_bank

    def optimiser(delvs):
        c_states, d1_states, d2_states, delv_bank = correct_orbit(delvs)

        c_final = state=c_states[-1]
        d1_final = orbits.ECI_Sat(d1_states[-1,:3],d1_states[-1,3:],times[n_burns,-1],ref).to_Baseline(state=c_final).state
        d2_final = orbits.ECI_Sat(d2_states[-1,:3],d2_states[-1,3:],times[n_burns,-1],ref).to_Baseline(state=c_final).state

        d1_true = orbits.init_deputy(ref,1,time=times[n_burns,-1],ref_orbit=False,state=c_final).to_Baseline(state=c_final).state
        d2_true = orbits.init_deputy(ref,2,time=times[n_burns,-1],ref_orbit=False,state=c_final).to_Baseline(state=c_final).state

        PHI = cost_function(d1_final-d1_true, d2_final - d2_true, d1_final, d2_final, np.sum(delv_bank,axis=0))
        #print(PHI)

        return PHI

    delvs = np.zeros((n_burns,2,3))
    x = minimize(optimiser,delvs,method="Nelder-Mead")
    c_states, d1_states, d2_states, delv_bank = correct_orbit(x.x)

    c_final = state=c_states[-1]
    d1_final = orbits.ECI_Sat(d1_states[-1,:3],d1_states[-1,3:],times[n_burns,-1],ref).to_Baseline(state=c_final).state
    d2_final = orbits.ECI_Sat(d2_states[-1,:3],d2_states[-1,3:],times[n_burns,-1],ref).to_Baseline(state=c_final).state

    d1_true = orbits.init_deputy(ref,1,time=times[n_burns,-1],ref_orbit=False,state=c_final).to_Baseline(state=c_final).state
    d2_true = orbits.init_deputy(ref,2,time=times[n_burns,-1],ref_orbit=False,state=c_final).to_Baseline(state=c_final).state

    print(d1_final-d1_true)
    print(d2_final-d2_true)

    return c_states, d1_states, d2_states, delv_bank, times.reshape(np.size(times))

#Initial states of the satellites
c0 = orbits.init_chief(ref).state
d10 = orbits.init_deputy(ref,1).state
d20 = orbits.init_deputy(ref,2).state

n_orbits = 2
period = ref.periodNC

for ix in range(n_orbits):

    print("Beginning orbit %s"%ix)

    t0 = ix*period
    t_final = (ix+0.5)*period

    c, d1, d2, times = propagate_integration(ref, c0, d10, d20, t0, t_final)

    t02 = times[-1]
    t_end = (ix+1)*period
    n_burns = 2
    burn_times = [t02, t02 + 10*60, t_end - 10*60, t_end]

    c2, d12, d22, delv2, tbank2 = recharge_fix(ref, c[-1], d1[-1], d2[-1], n_burns, burn_times)

    print("Delv for orbit fix %s: "%ix + str(np.sum(delv2,axis=0)))

    c0 = c2[-1]
    d10 = d12[-1]
    d20 = d22[-1]
