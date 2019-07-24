from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits as orbits
from matplotlib.collections import LineCollection
from modules.Schweighart_J2_solved import equations_creation
from multiprocessing import Pool

def worker(params):
    alt,delta_r_max,inc_0,Om_0,ra,dec = params

    R_e = const.R_earth.value  #In m
    #Orbital radius the sum of earth radius and altitude
    R_orb = R_e + alt

    #------------------------------------------------------------------------------------------
    #Calculate reference orbit, in the geocentric (ECI) frame (See Orbit module)
    ref = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

    #Number of orbits
    n_orbits = 1
    #Number of phases in each orbit
    n_phases = 1000
    #Total evaluation points
    n_times = int(n_orbits*n_phases)
    times = np.linspace(0,ref.period*n_orbits,n_times) #Create list of times

    #Initial reference orbit state
    pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(0)

    #Initial states of the satellites
    chief_0 = orbits.init_chief(ref,0).to_LVLH(pos_ref,vel_ref,LVLH)
    deputy1_0 = orbits.init_deputy(ref,0,1).to_LVLH(pos_ref,vel_ref,LVLH)
    deputy2_0 = orbits.init_deputy(ref,0,2).to_LVLH(pos_ref,vel_ref,LVLH)

    #Create the state equations, from t = 0
    base_equation = equations_creation(ref)
    chief_equation_0 = base_equation(0,chief_0.state)
    deputy1_equation_0 = base_equation(0,deputy1_0.state)
    deputy2_equation_0 = base_equation(0,deputy2_0.state)

    chief_p_states = chief_equation_0(times).transpose()
    deputy1_p_states = deputy1_equation_0(times).transpose()
    deputy2_p_states = deputy2_equation_0(times).transpose()

    d1_rel_states = deputy1_p_states - chief_p_states
    d2_rel_states = deputy2_p_states - chief_p_states

    ECI_rc = np.zeros((len(times),3))
    rel_p_dep1 = []
    rel_p_dep2 = []

    print("Integration Done")
    for i in range(len(times)):
        pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(times[i],True)
        rel_p_dep1.append(orbits.LVLH_Sat(d1_rel_states[i,:3],d1_rel_states[i,3:],times[i],ref).to_Baseline(LVLH,Base))
        rel_p_dep2.append(orbits.LVLH_Sat(d2_rel_states[i,:3],d2_rel_states[i,3:],times[i],ref).to_Baseline(LVLH,Base))
    print("Classifying Done")

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
        s_hat_sep[ix] = s_hat_drd1[ix] - s_hat_drd2[ix]

        #Sum of the separation along the star direction and the baseline direction
        total_sep[ix] = baseline_sep[ix] + s_hat_sep[ix]

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
    acc_delta_s = np.abs(acc(s_hat_sep,times))
    acc_total = np.abs(acc(total_sep,times))

    #Maximum accelerations
    max_acc_s1 = max(acc_s1)
    max_acc_s2 = max(acc_s2)
    max_acc_delta_b = max(acc_delta_b)
    max_acc_delta_s = max(acc_delta_s)
    max_acc_total = max(acc_total)

    #Delta v (Integral of the absolute value of the acceleration)
    delta_v_s1 = np.trapz(acc_s1)
    delta_v_s2 = np.trapz(acc_s2)
    delta_v_delta_b = np.trapz(acc_delta_b)
    delta_v_delta_s = np.trapz(acc_delta_s)
    delta_v_total = np.trapz(acc_total)

    #Result array
    #result[0] is the max a between deputy 1 and chief in the star direction
    #result[1] is the max a between deputy 2 and chief in the star direction
    #result[2] is the max a between the two deputies in the star direction (ie the difference between 0 and 1)
    #result[3] is the max a in the baseline direction
    #result[4] is the max total a (sum of 2 and 3; the total acceleration that needs to be corrected for)
    #result[5-9] are the same, but for delta v

    result = np.array([alt,delta_r_max,inc_0,Om_0,ra,dec,
                       max_acc_s1,max_acc_s2,max_acc_delta_s,
                       max_acc_delta_b,max_acc_total,delta_v_s1,delta_v_s2,
                       delta_v_delta_s,delta_v_delta_b,delta_v_total])

    return

inputs = [(alt,delta_r_max,np.radians(inc_0),np.radians(Om_0),np.radians(ra),
           np.radians(dec)) for delta_r_max in [0.3e3,0.6e3] for alt in [5e5,1e6]
           for inc_0 in range(-90,90,90) for Om_0 in range(0,360,180)
           for ra in range(0,360,180) for dec in range(-90,90,90)]

p = Pool(processes=2)
result = p.map(worker,inputs)
result = np.array(result)
np.save("variance.npy",result)
p.close()
