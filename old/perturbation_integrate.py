import numpy as np
from orbits import sat_orbit
import astropy.constants as const
from scipy.integrate import solve_ivp

""" Differential equation function with NO J2 perturbation """
def none_dX_dt(t, state, LVLH_orbit):
    r = state[:3] #Position
    v = state[3:] #Velocity

    dX0 = v[0]
    dX1 = v[1]
    dX2 = v[2]

    n = LVLH_orbit.ang_vel #Angular velocity
    mu = const.GM_earth.value #Graviational parameter
    omega = np.array([0,0,n]) #Angular velocity vector in LVLH frame
    
    print(t)
    
    #HCW Equations - Until this works, will use the analytical form
    K = np.diag(np.array([3*n**2,0,-(n**2)]))
    #Gamma2 = n**2/LVLH_orbit.R_orb*np.array([-3*r[0]**2 + 1.5*r[1]**2 + 1.5*r[2]**2, 3*r[0]*r[1], 3*r[0]*r[2]])
    a = -2*np.cross(omega,v) + np.matmul(K,r)# + Gamma2
    
    #Position vector of deputy
    #rd = np.array([LVLH_orbit.R_orb+r[0],r[1],r[2]])
    #Acceleration vector - analytical version (See Butcher 18)
    # a = -2*np.cross(omega,v) - np.cross(omega,np.cross(omega,rd)) - mu*np.array([-2*r[0],r[1],r[2]])/np.linalg.norm(rd)**3
    
    dX3 = a[0]
    dX4 = a[1]
    dX5 = a[2]
    return [dX0,dX1,dX2,dX3,dX4,dX5]

""" Differential equation function with J2 perturbation NOT READY!!!!!
def J2_dX_dt(t, state, LVLH_orbit):
    r = state[:3] #Position
    v = state[3:] #Velocity

    dX0 = v[0]
    dX1 = v[1]
    dX2 = v[2]

    n = LVLH_orbit.ang_vel #

    perturb_acc = f(LVLH_orbit)

    K = np.diag(np.array([3*n**2,0,-(n**2)]))
    Gamma2 = n**2/LVLH_orbit.R_orb*np.array([-3*r[0]**2 + 1.5*r[1]**2 + 1.5*r[2]**2, 3*r[0]*r[1], 3*r[0]*r[2]])

    a = -2*np.cross(omega,v) + np.matmul(K,r) + Gamma2 + perturb_acc

    dX3 = a[0]
    dX4 = a[1]
    dX5 = a[2]
    return [dX0,dX1,dX2,dX3,dX4,dX5]

"""

def perturb_orbit(LVLH_orbit,perturb_func):
    p_orbit = sat_orbit(LVLH_orbit.n_p,LVLH_orbit.R_orb) #New Orbit

    tspan = LVLH_orbit.phase/LVLH_orbit.ang_vel #Array of times
    X0_d1 = LVLH_orbit.deputy1_sep_state_vec()[:,0] #Initial state vector for deputy 1
    X0_d2 = LVLH_orbit.deputy2_sep_state_vec()[:,0] #Initial state vector for deputy 2
    
    rtol = 1e-6
    atol = 1e-12

    #Integrate the orbits
    X_d1 = solve_ivp(lambda t, y: perturb_func(t,y,LVLH_orbit), [tspan[0],tspan[-1]], X0_d1, t_eval = tspan, rtol = rtol, atol = atol, max_step=1)
    X_d2 = solve_ivp(lambda t, y: perturb_func(t,y,LVLH_orbit), [tspan[0],tspan[-1]], X0_d2, t_eval = tspan, rtol = rtol, atol = atol, max_step=1)

    #Positions and velocities of the peturbed orbit
    p_orbit.chief_pos = LVLH_orbit.chief_pos
    p_orbit.chief_vel = LVLH_orbit.chief_vel
    p_orbit.deputy1_pos_sep = np.transpose(X_d1.y[:3])
    p_orbit.deputy1_vel_sep = np.transpose(X_d1.y[3:])
    p_orbit.deputy2_pos_sep = np.transpose(X_d2.y[:3])
    p_orbit.deputy2_vel_sep = np.transpose(X_d2.y[3:])
    p_orbit.deputy1_pos = p_orbit.deputy1_pos_sep + p_orbit.chief_pos
    p_orbit.deputy1_vel = p_orbit.deputy1_vel_sep + p_orbit.chief_vel
    p_orbit.deputy2_pos = p_orbit.deputy2_pos_sep + p_orbit.chief_pos
    p_orbit.deputy2_vel = p_orbit.deputy2_vel_sep + p_orbit.chief_vel

    return p_orbit
