import numpy as np
import astropy.constants as const
from scipy.optimize import fsolve
import quaternions as qt

""" J2 Perturbation function from Schweighart's paper """
def J2_pet(state0,ECI,rotation):

    r_ref = ECI.R_orb
    i_ref = ECI.inc_0

    h_dep = qt.rotate(ECI.h_0,rotation)
    i_dep = np.arccos(h_dep[2]/np.linalg.norm(h_dep))
    print(i_dep)

    J2 = 0.00108263
    R_e = const.R_earth.value
    mu = const.GM_earth.value

    [x_0,y_0,z_0] = state0[:3]
    dz_0 = state0[5]
    s = 3*J2*R_e**2/(8*r_ref**2)*(1+3*np.cos(2*i_ref))
    c = np.sqrt(1+s)
    n = ECI.ang_vel
    k = n*c+3*n*J2*R_e**2/(2*r_ref**2)*np.cos(i_ref)**2
    i_sat1 = dz_0/(k*r_ref)+i_dep
    omega_0 = z_0/(r_ref*np.sin(i_ref))
    gamma_0 = np.pi/2 - np.arctan((1/np.tan(i_dep)*np.sin(i_sat1)-np.cos(i_sat1)*np.cos(omega_0))/np.sin(omega_0))
    phi_0 = np.arccos(np.cos(i_sat1)*np.cos(i_dep)+np.sin(i_sat1)*np.sin(i_dep)*np.cos(omega_0))
    d_omega_sat1 = -3*n*J2*R_e**2/(2*r_ref**2)*np.cos(i_sat1)
    d_omega_dep = -3*n*J2*R_e**2/(2*r_ref**2)*np.cos(i_dep)
    q = n*c - (np.cos(gamma_0)*np.sin(gamma_0)*1/np.tan(omega_0)-np.sin(gamma_0)**2*np.cos(i_sat1))*(d_omega_sat1 - d_omega_dep)-d_omega_sat1*np.cos(i_sat1)
    l = -r_ref*np.sin(i_sat1)*np.sin(i_dep)*np.sin(omega_0)/np.sin(phi_0)*(d_omega_sat1-d_omega_dep)

    def equations(p):
        m,phi = p
        return(m*np.sin(phi)-z_0,l*np.sin(phi)+q*m*np.cos(phi)-dz_0)

    m,phi = fsolve(equations,(1,1))

    def J2_pet_func(t,state):
        [x,y,z] = state[:3] #Position
        [dx,dy,dz] = state[3:] #Velocity
        print(t)
        dX0 = dx
        dX1 = dy
        dX2 = dz

        dX3 = 2*n*c*dy + (5*c**2-2)*n**2*x
        dX4 = -2*n*c*dx
        dX5 = -q**2*z + 2*l*q*np.cos(q*t+phi)
        return np.array([dX0,dX1,dX2,dX3,dX4,dX5])

    return J2_pet_func
