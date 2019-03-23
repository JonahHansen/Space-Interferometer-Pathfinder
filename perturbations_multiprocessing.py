from __future__ import print_function
import numpy as np
import astropy.constants as const
from scipy.integrate import solve_ivp
from orbits import ECI_orbit
from perturbations import dX_dt
from itertools import product
from multiprocessing import Pool
import json

param_ls = []

n_inc = 360
n_dec = 360

inc_0 = np.radians(np.linspace(0,90,n_inc))
Om_0 = np.radians(np.array([0]))
ra = np.radians(np.array([0,90]))
dec = np.radians(np.linspace(-90,90,n_dec))
R_orb = np.array([500e3,1000e3])+const.R_earth.value
delta_r_max = np.array([0.3e3,0.1e3])

param_ls = list(product(R_orb,delta_r_max,inc_0,Om_0,ra,dec))

inputs = list(enumerate(param_ls))

input_len = len(inputs)

#------------------------------------------------------------------------------------------
def worker(arg):

    index = arg[0]
    params = arg[1]

    print(index/input_len*100)

    #Calculate orbit, in the geocentric (ECI) frame
    ECI = ECI_orbit(*params)

    num_times = 1000
    times = np.linspace(0,ECI.period,num_times) #Create list of times

    """Initialise arrays"""
    ECI_rc = np.zeros((num_times,6)) #Chief ECI position vector
    s_hats = np.zeros((num_times,3)) #Star vectors

    i = 0
    for t in times:
        ECI_rc[i] = ECI.chief_state(t)
        rot_mat = ECI.to_LVLH_mat(ECI_rc[i]) #Rotation matrix
        s_hats[i] = np.dot(rot_mat,ECI.s_hat) #Star vectors
        i += 1

    rot_mat = ECI.to_LVLH_mat(ECI_rc[0])
    ECI_rd1_0 = ECI.deputy1_state(ECI_rc[0]) #Deputy 1 position
    ECI_rd2_0 = ECI.deputy2_state(ECI_rc[0]) #Deputy 2 position
    LVLH_drd1_0 = ECI.to_LVLH_state(ECI_rc[0],rot_mat,ECI_rd1_0) #Initial LVLH separation state for deputy 1
    LVLH_drd2_0 = ECI.to_LVLH_state(ECI_rc[0],rot_mat,ECI_rd2_0) #Initial LVLH separation state for deputy 1

    rtol = 1e-6
    atol = 1e-12
    step = 100

    #Integrate the orbits
    X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ECI), [times[0],times[-1]], LVLH_drd1_0, t_eval = times, rtol = rtol, atol = atol, max_step=step)
    X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ECI), [times[0],times[-1]], LVLH_drd2_0, t_eval = times, rtol = rtol, atol = atol, max_step=step)

    #Peturbed orbits
    pert_LVLH_drd1 = np.transpose(X_d1.y)
    pert_LVLH_drd2 = np.transpose(X_d2.y)

    #--------------------------------------------------------------------------------------------- #

    baseline_sep = np.zeros(num_times) #Separation along the baseline
    s_hat_drd1 = np.zeros(num_times) #Deputy1 position in star direction
    s_hat_drd2 = np.zeros(num_times) #Deputy2 position in star direction
    s_hat_sep = np.zeros(num_times) #Separation along the baseline
    total_sep = np.zeros(num_times) #Total separation

    for ix in range(num_times):
        baseline_sep[ix] = np.linalg.norm(pert_LVLH_drd1[ix]) - np.linalg.norm(pert_LVLH_drd2[ix])
        s_hat_drd1[ix] = np.dot(pert_LVLH_drd1[ix,:3],s_hats[ix])
        s_hat_drd2[ix] = np.dot(pert_LVLH_drd2[ix,:3],s_hats[ix])
        s_hat_sep[ix] = s_hat_drd1[ix] - s_hat_drd2[ix]
        total_sep[ix] = baseline_sep[ix] + s_hat_sep[ix]

    #Numerical differentiation
    def acc(pos):
        vel = np.gradient(pos, edge_order=2)
        acc = np.gradient(vel, edge_order=2)
        return np.abs(acc)

    max_sep_s1 = max(abs(s_hat_drd1))
    max_sep_s2 = max(abs(s_hat_drd2))
    max_sep_delta_b = max(abs(baseline_sep))
    max_sep_delta_s = max(abs(s_hat_sep))
    max_sep_total = max(abs(total_sep))

    #Accelerations
    acc_s1 = np.abs(acc(s_hat_drd1))
    acc_s2 = np.abs(acc(s_hat_drd2))
    acc_delta_b = np.abs(acc(baseline_sep))
    acc_delta_s = np.abs(acc(s_hat_sep))
    acc_total = np.abs(acc(total_sep))

    #Maximum accelerations
    max_acc_s1 = max(acc_s1)
    max_acc_s2 = max(acc_s2)
    max_acc_delta_b = max(acc_delta_b)
    max_acc_delta_s = max(acc_delta_s)
    max_acc_total = max(acc_total)

    #Delta v
    delta_v_s1 = np.trapz(acc_s1)
    delta_v_s2 = np.trapz(acc_s2)
    delta_v_delta_b = np.trapz(acc_delta_b)
    delta_v_delta_s = np.trapz(acc_delta_s)
    delta_v_total = np.trapz(acc_total)


    output_dict = {"R_orb":params[0], "Delta_r":params[1], "inc_0":params[2], "Om_0":params[3],
                   "ra":params[4], "dec":params[5], "Max_sep_s1": max_sep_s1, "Max_sep_s2": max_sep_s2,
                   "Max_sep_delta_s": max_sep_delta_s, "Max_sep_delta_b": max_sep_delta_b,
                   "Max_sep_total": max_sep_total, "Max_a_s1": max_acc_s1, "Max_a_s2": max_acc_s2,
                   "Max_a_delta_s": max_acc_delta_s, "Max_a_delta_b": max_acc_delta_b,
                   "Max_a_total": max_acc_total, "Delta_v_s1": delta_v_s1, "Delta_v_s2": delta_v_s2,
                   "Delta_v_delta_s": delta_v_delta_s, "Delta_v_delta_b": delta_v_delta_b,
                   "Delta_v_total": delta_v_total}
    return output_dict

p = Pool(processes=25)

result = p.map(worker,inputs)

with open('bigboy.json', 'w') as f:  # writing JSON object
     json.dump(result, f)
