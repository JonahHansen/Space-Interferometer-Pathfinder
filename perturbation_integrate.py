import numpy as np
from orbits import sat_orbit
from scipy.integrate import solve_ivp

def none_dX_dt(t, state, LVLH_orbit):
    r = state[:3]
    v = state[3:]

    dX0 = v[0]
    dX1 = v[1]
    dX2 = v[2]

    n = LVLH_orbit.ang_vel
    mu = n**2*LVLH_orbit.R_orb**3
    omega = np.array([0,0,n])

    K = np.diag(np.array([3*n**2,0,-(n**2)]))
    Gamma2 = n**2/LVLH_orbit.R_orb*np.array([-3*r[0]**2 + 1.5*r[1]**2 + 1.5*r[2]**2, 3*r[0]*r[1], 3*r[0]*r[2]])

    a = -2*np.cross(omega,v) + np.matmul(K,r)# + Gamma2
    #rd = np.array([LVLH_orbit.R_orb+r[0],r[1],r[2]])
    #a = -2*np.cross(omega,v) - np.cross(omega,np.cross(omega,rd)) - mu*rd/np.linalg.norm(rd)**3
    dX3 = a[0]
    dX4 = a[1]
    dX5 = a[2]
    return [dX0,dX1,dX2,dX3,dX4,dX5]

def J2_dX_dt(t, state, LVLH_orbit):
    r = state[:3]
    v = state[3:]

    dX0 = v[0]
    dX1 = v[1]
    dX2 = v[2]

    n = LVLH_orbit.ang_vel

    perturb_acc = f(LVLH_orbit)

    K = np.diag(np.array([3*n**2,0,-(n**2)]))
    Gamma2 = n**2/LVLH_orbit.R_orb*np.array([-3*r[0]**2 + 1.5*r[1]**2 + 1.5*r[2]**2, 3*r[0]*r[1], 3*r[0]*r[2]])

    a = -2*np.cross(omega,v) + np.matmul(K,r) + Gamma2 + perturb_acc

    dX3 = a[0]
    dX4 = a[1]
    dX5 = a[2]
    return [dX0,dX1,dX2,dX3,dX4,dX5]

def perturb_orbit(LVLH_orbit,perturb_func):
    p_orbit = sat_orbit(LVLH_orbit.n_p,LVLH_orbit.R_orb)

    tspan = [np.min(LVLH_orbit.phase),np.max(LVLH_orbit.phase)]
    X0_d1 = LVLH_orbit.deputy1_sep_state_vec()[:,0]
    X0_d2 = LVLH_orbit.deputy2_sep_state_vec()[:,0]

    print(X0_d1)

    X_d1 = solve_ivp(lambda t, y: perturb_func(t,y,LVLH_orbit), tspan, X0_d1, t_eval = LVLH_orbit.phase)
    X_d2 = solve_ivp(lambda t, y: perturb_func(t,y,LVLH_orbit), tspan, X0_d2, t_eval = LVLH_orbit.phase)

    p_orbit.deputy1_pos_sep = np.transpose(X_d1.y[:3])
    p_orbit.deputy1_vel_sep = np.transpose(X_d1.y[3:])
    p_orbit.deputy2_pos_sep = np.transpose(X_d2.y[:3])
    p_orbit.deputy2_vel_sep = np.transpose(X_d2.y[3:])

    return p_orbit
