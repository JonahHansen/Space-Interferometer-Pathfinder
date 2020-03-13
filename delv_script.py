""" SCRIPT TO TEST ORBIT CORRECTIONS AND DELTA V REQUIREMENTS """
""" Takes in inc, Om, ra, dec from the command line """

import numpy as np
import astropy.constants as const
import modules.orbits as orbits
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, root, brentq
import sys

#Set up orbital configuration

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

#Calculate reference orbit, in the geocentric (ECI) frame (See Orbit module)
ref = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)


######################## J2 Propagation #################################

""" J2 Perturbation acceleration function
    Inputs:
        t - time
        state - state of satellite
        ref - reference orbit
    Output: Acceleration at time t
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


""" Propagates the orbit using the ECI J2 integrator
    Inputs:
        X0 - initial state
        times - list of times for evaluation
        ref - reference orbit
    Output: list of states at the given times
"""
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


###################### Integration Correction ############################

""" Calculate the max delta v required during the integration times
    Inputs:
        c_states - list of chief states
        d1_states - list of deputy1 states
        d2_states - list of deputy2 states
        times - list of times
        ref - reference orbit
    Output: array of max delta v [chief, deputy1, deputy2]
"""
def calc_delv(c_states,d1_states,d2_states,times,ref):
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

    delv_deputy = 2*delta_v_s1 + delta_v_delta_b
    delv_chief = delta_v_s1

    return np.array([delv_chief, delv_deputy, delv_deputy])


""" Function that propagates the spacecraft from the start of integration
    to the end of integration, and then forces the star separation to be zero
    Inputs:
        c0 - initial chief state
        d10 - initial deputy1 state
        d20 - initial deputy2 states
        t0 - starting time
        t_final - end time
        ref - reference orbit
    Outputs:
        c_states - list of chief states
        d1_states - list of deputy1 states (s direction set to 0)
        d2_states - list of deputy2 states (s direction set to 0)
        delv - array of max delta v required to fix during integration [chief, deputy1, deputy2]
"""
def propagate_integration(c0, d10, d20, t0, t_final, ref):

    #List of times
    times = np.linspace(t0,t_final,500)

    #propagate the satellites to the end of integration
    c_states = propagate_orbit(c0,times,ref)
    d1_states = propagate_orbit(d10,times,ref)
    d2_states = propagate_orbit(d20,times,ref)

    #Calculate maximum delta v
    delv = calc_delv(c_states,d1_states,d2_states,times,ref)

    #Convert deputies to baseline frame
    d1sat = orbits.ECI_Sat(d1_states[-1,:3],d1_states[-1,3:],times[-1],ref).to_Baseline(state=c_states[-1])
    d2sat = orbits.ECI_Sat(d2_states[-1,:3],d2_states[-1,3:],times[-1],ref).to_Baseline(state=c_states[-1])

    #Force s direction to be 0
    d1sat.pos[2] = 0
    d2sat.pos[2] = 0

    #Convert back to ECI frame
    d1_states[-1] = d1sat.to_ECI(state=c_states[-1]).state
    d2_states[-1] = d2sat.to_ECI(state=c_states[-1]).state

    return c_states,d1_states,d2_states, delv


######################## Recharge correction ##################################

""" Cost function for optimiser
    Inputs:
        d1 - final state vector of deputy1
        d2 - final the state vector of deputy1
        ideal_d1 - ideal end state of deputy 1
        ideal_d2 - ideal end state of deputy 2
        delv - list of total delv used in thrusting [chief, deputy1, deputy2]
    Outputs:
        Phi - cost variable
"""
def cost_function(d1, d2, ideal_d1, ideal_d2, delv):

    #Calculate residuals
    sig_state1 = d1 - ideal_d1
    sig_state2 = d2 - ideal_d2

    #Weights for the state residuals
    kappa_d1 = 1000*np.array([1,1,3,2,2,3])
    kappa_d2 = 1000*np.array([1,1,3,2,2,3])

    #Weights for the delta v components
    kappa_dv = np.array([1,1,1])

    Phi = np.dot(kappa_d1,sig_state1**2) + np.dot(kappa_d1,sig_state2**2) + np.dot(kappa_dv,delv**2)
    return Phi


""" Root function for solver
    Inputs:
        d1 - final state vector of deputy1
        d2 - final the state vector of deputy1
        ideal_d1 - ideal end state of deputy 1
        ideal_d2 - ideal end state of deputy 2
    Output:
        Phi - variable to reduce down to zero
"""
def root_function(d1, d2, ideal_d1, ideal_d2):

    #Calculate residuals
    sig_state1 = d1 - ideal_d1
    sig_state2 = d2 - ideal_d2


    Phi = np.append(sig_state1,sig_state2)
    return Phi


""" Calculate the states of the satellites for the rest of the orbit
    by calculating the optimum thrusts at a list of given times

    Input:
        c0 - initial chief state
        d10 - initial deputy1 state
        d20 - initial deputy2 state
        burn_times - List of times of thrusts, starting with initial time and ending
                     with final time. ie. [t0, thrust 1, thrust 2, t_final]
        ref - reference orbit

    Output:
        c_states - list of chief states
        d1_states - list of deputy1 states (s direction set to 0)
        d2_states - list of deputy2 states (s direction set to 0)
        delv - array of max delta v required to fix during integration [chief, deputy1, deputy2]
"""
def recharge_fix(c0,d10,d20,burn_times,ref):

    #Number of burns is the number of middle elements in the array
    n_burns = len(burn_times)-2

    #List of times, broken up between burns
    times = np.zeros((n_burns+1,50))

    for i in range(n_burns+1):
        times[i] = np.linspace(burn_times[i],burn_times[i+1],50)


    #Initialise variables
    c_states_fin = np.zeros((50*(n_burns+1),6))
    d1_states_fin = np.zeros((50*(n_burns+1),6))
    d2_states_fin = np.zeros((50*(n_burns+1),6))
    delv_bank_fin = np.zeros((n_burns,3))


    #Propagate the satellites until the thrust time
    c_states_part = propagate_orbit(c0,times[0],ref)
    d1_states_part = propagate_orbit(d10,times[0],ref)
    d2_states_part = propagate_orbit(d20,times[0],ref)

    #Save into array
    c_states_fin[:50] = c_states_part
    d1_states_fin[:50] = d1_states_part
    d2_states_fin[:50] = d2_states_part

    """ Function to return the states and delta v of the spacecraft
        given a list of delta v burns at the times set
        by the variable "burn times"

        Currently NOT thrusting chief satellite
    """
    def correct_orbit(delvs):

        #Initialise variables
        c_states = np.zeros((50*(n_burns),6))
        d1_states = np.zeros((50*(n_burns),6))
        d2_states = np.zeros((50*(n_burns),6))
        delv_bank = np.zeros((n_burns,3))


        #Reshape (due to the optimiser)
        delvs = delvs.reshape((n_burns,2,3))

        #Delta v for first thrust
        delv_c = 0
        delv_d1 = delvs[0,0]
        delv_d2 = delvs[0,1]

        #Save delta v into array
        delv_bank[0] = np.array([np.linalg.norm(delv_c),np.linalg.norm(delv_d1),np.linalg.norm(delv_d2)])

        #Add delta v to the final state and loop
        c = c_states_fin[49]# + np.append(np.zeros(3),delv_c)
        d1 = d1_states_fin[49] + np.append(np.zeros(3),delv_d1)
        d2 = d2_states_fin[49] + np.append(np.zeros(3),delv_d2)

        for i in range(0,n_burns-1):

            #Propagate the satellites until the thrust time
            c_states_part = propagate_orbit(c,times[i+1],ref)
            d1_states_part = propagate_orbit(d1,times[i+1],ref)
            d2_states_part = propagate_orbit(d2,times[i+1],ref)

            #Save into array
            c_states[i*50:i*50+50] = c_states_part
            d1_states[i*50:i*50+50] = d1_states_part
            d2_states[i*50:i*50+50] = d2_states_part

            #Delta v for current thrust
            delv_c = 0
            delv_d1 = delvs[i+1,0]
            delv_d2 = delvs[i+1,1]

            #Save delta v into array
            delv_bank[i+1] = np.array([np.linalg.norm(delv_c),np.linalg.norm(delv_d1),np.linalg.norm(delv_d2)])

            #Add delta v to the final state and loop
            c = c_states[50+i*50-1]# + np.append(np.zeros(3),delv_c)
            d1 = d1_states[50+i*50-1] + np.append(np.zeros(3),delv_d1)
            d2 = d2_states[50+i*50-1] + np.append(np.zeros(3),delv_d2)

        #Propagate the satellites after the final thrust
        c_states_part = propagate_orbit(c,times[n_burns],ref)
        d1_states_part = propagate_orbit(d1,times[n_burns],ref)
        d2_states_part = propagate_orbit(d2,times[n_burns],ref)

        #Save to array and return the array of states and delta v
        c_states[(n_burns-1)*50:(n_burns-1)*50+50] = c_states_part
        d1_states[(n_burns-1)*50:(n_burns-1)*50+50] = d1_states_part
        d2_states[(n_burns-1)*50:(n_burns-1)*50+50] = d2_states_part

        return c_states, d1_states, d2_states, delv_bank


    """ Optimiser (Root finder) function
        Takes in a list of delta vs and attempts to minimise (solve) using the cost (root) function
    """
    def optimiser(delvs):

        #Given the delta vs, calculate the states
        c_states, d1_states, d2_states, delv_bank = correct_orbit(delvs)

        #Final chief state
        c_final = state=c_states[-1]

        #Final deputy states in the baseline frame
        d1_final = orbits.ECI_Sat(d1_states[-1,:3],d1_states[-1,3:],times[n_burns,-1],ref).state#.to_Baseline(state=c_final).state
        d2_final = orbits.ECI_Sat(d2_states[-1,:3],d2_states[-1,3:],times[n_burns,-1],ref).state#.to_Baseline(state=c_final).state

        #Ideal deputy states (where the deputies should be given the position of the chief)
        d1_ideal = orbits.init_deputy(ref,1,time=times[n_burns,-1],ref_orbit=False,state=c_final).state#.to_Baseline(state=c_final).state
        d2_ideal = orbits.init_deputy(ref,2,time=times[n_burns,-1],ref_orbit=False,state=c_final).state#.to_Baseline(state=c_final).state

        #Cost function calculation
        #Phi = cost_function(d1_final, d2_final, d1_ideal, d2_ideal, np.sum(delv_bank,axis=0))
        
        #Root function calculation
        Phi = root_function(d1_final, d2_final, d1_ideal, d2_ideal)
        #print(Phi)

        return Phi

    #Solve for the thrusts.
    x = root(optimiser,np.zeros((n_burns,2,3)),method='hybr')

    #Calculate states based on optimum thrusts
    c_states, d1_states, d2_states, delv_bank = correct_orbit(x.x)

    c_states_fin[50:] = c_states
    d1_states_fin[50:] = d1_states
    d2_states_fin[50:] = d2_states

    #Calculate residuals to print to terminal:
    #Final chief state
    c_final = state=c_states[-1]

    #Final deputy states in the baseline frame
    d1_final = orbits.ECI_Sat(d1_states[-1,:3],d1_states[-1,3:],times[n_burns,-1],ref).to_Baseline(state=c_final).state
    d2_final = orbits.ECI_Sat(d2_states[-1,:3],d2_states[-1,3:],times[n_burns,-1],ref).to_Baseline(state=c_final).state

    #Ideal deputy states (where the deputies should be given the position of the chief)
    d1_ideal = orbits.init_deputy(ref,1,time=times[n_burns,-1],ref_orbit=False,state=c_final).to_Baseline(state=c_final).state
    d2_ideal = orbits.init_deputy(ref,2,time=times[n_burns,-1],ref_orbit=False,state=c_final).to_Baseline(state=c_final).state

    print("Residuals for Deputy 1: " + str(np.abs(d1_final-d1_ideal)))
    print("Residuals for Deputy 2: " + str(np.abs(d2_final-d2_ideal)))

    return c_states_fin, d1_states_fin, d2_states_fin, delv_bank


################################### Script part ##############################################

if __name__ == "__main__":
    #Initial states of the satellites
    c0 = orbits.init_chief(ref).state
    d10 = orbits.init_deputy(ref,1).state
    d20 = orbits.init_deputy(ref,2).state
    
    #Number of orbits to calculate
    n_orbits = 1
    
    #What period to use? NC/K for Schweighart correction
    period = ref.periodK
    
    total_delv = np.zeros((n_orbits,3))
    
    for ix in range(n_orbits):
    
        print("Beginning orbit %s"%ix)
    
        #Start and end of integration
        t0 = ix*period
        t_final = (ix+0.5)*period
    
        #Propagate through integration, forcing s direction to be 0
        c, d1, d2, delv = propagate_integration(c0, d10, d20, t0, t_final, ref)
    
        #Half way to end of orbit
        t02 = t_final
        t_end = (ix+1)*period
    
        #Middle times are the times at which a thrust will occur (in s).
        #i.e an array of length 4 has two thrusts...
        burn_times = [t02, t02 + 10*60, t_end - 10*60, t_end]
    
    
        #Correct the orbit through optimising the thrust
        c2, d12, d22, delv2 = recharge_fix(c[-1], d1[-1], d2[-1], burn_times, ref)
    
        #Print the delv used per satellite in orbit correction
        print("Delv for integration fix %s: "%ix + str(delv))
        print("Delv for recharge fix %s: "%ix + str(np.sum(delv2,axis=0)))
        print("Total Delv for orbit fix %s: "%ix + str(np.sum(delv2,axis=0)+delv))
    
        total_delv[ix] = np.sum(delv2,axis=0)+delv
    
        #New initial states
        c0 = c2[-1]
        d10 = d12[-1]
        d20 = d22[-1]
