import numpy as np
import orbits
from scipy.integrate import solve_ivp

def dX_dt(t, state, LVLH_orbit, f):
    r = state[:3]
    v = state[3:]
    
    dX0 = v[0]
    dX1 = v[1]
    dX2 = v[2]
    
    n = LVLH_orbit.ang_vel
    
    perturb_acc = f(LVLH_orbit)
    
    K = np.diag(3*n**2,0,-(n**2))
    Gamma2 = n**2/LVLH_orbit.R_orb*np.array([-3*r[0]**2 + 1.5*r[1]**2 + 1.5*r[2]**2, 3*r[0]*r[1], 3*r[0]*r[2]])
    
    a = -2*np.cross(omega,v) + np.matmul(K,r) + Gamma2 + perturb_acc
    
    dX3 = a[0]
    dX4 = a[1]
    dX5 = a[2]
    return [dX0,dX1,dX2,dX3,dX4,dX5]
    
def perturb_orbit(LVLH_orbit,perturb_func):
    p_orbit = orbits.sat_orbit(LVLH_orbit.n_p,LVLH_orbit.R_orb)
    
    tspan = [np.min(LVLH_orbit.phase),np.max(LVLH_orbit.phase)]
    X0_d1 = LVLH_orbit.deputy1_state_vec()[0]
    X0_d2 = LVLH_orbit.deputy2_state_vec()[0]
    
    X_d1 = solve_ivp(fun = lambda t, y: dX_dt(t,y,LVLH_orbit,perturb_func), tspan, X0_d1, t_eval = LVLH_orbit.phase)
    X_d2 = solve_ivp(fun = lambda t, y: dX_dt(t,y,LVLH_orbit,perturb_func), tspan, X0_d2, t_eval = LVLH_orbit.phase)
    
    p_orbit.deputy1_pos = X_d1[:3]
    p_orbit.deputy1_vel = X_d1[3:]
    p_orbit.deputy2_pos = X_d2[:3]
    p_orbit.deputy2_vel = X_d2[3:]
    
    return p_orbit