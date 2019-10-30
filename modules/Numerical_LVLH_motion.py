import numpy as np
import astropy.constants as const
from scipy.optimize import fsolve
import modules.quaternions as qt

"""
J2 Perturbation function from Schweighart's paper.
Takes an initial satellite/ref orbit and returns a function
that calculates the state at any time t. Relative to reference orbit.
Inputs:
    sat0 - initial satellite class in LVLH frame
    ref - reference orbit
Output:
    J2_pert_func - an integrable function that propagates a satellite's position
"""
def J2_pert_num(sat0,ref):

    #DEFINE VARIABLES AS IN PAPER
    r_ref = ref.R_orb #Radius of the reference orbit (and chief)
    i_ref = ref.inc_0 #Inclination of the reference orbit (and chief)

    J2 = 0.00108263
    R_e = const.R_earth.value

    #Initial conditions
    [x_0,y_0,z_0] = sat0.pos
    dz_0 = sat0.vel[2]

    #Define variables from Schweghart
    c = ref.Sch_c
    n = ref.ang_vel
    k = ref.Sch_k

    i_sat = dz_0/(k*r_ref)+i_ref

    if i_ref == 0:
        omega_0 = 0
    else:
        omega_0 = z_0/(r_ref*np.sin(i_ref))

    if (omega_0 and i_ref) != 0:
        gamma_0 = np.arctan(1/((1/np.tan(i_ref)*np.sin(i_sat)-np.cos(i_sat)*np.cos(omega_0))/np.sin(omega_0)))
    else:
        gamma_0 = 0

    phi_0 = np.arccos(np.cos(i_sat)*np.cos(i_ref)+np.sin(i_sat)*np.sin(i_ref)*np.cos(omega_0))
    d_omega_sat = -3*n*J2*R_e**2/(2*r_ref**2)*np.cos(i_sat)
    d_omega_ref = -3*n*J2*R_e**2/(2*r_ref**2)*np.cos(i_ref)

    temp = np.cos(gamma_0)*np.sin(gamma_0)*1/np.tan(omega_0)
    temp = temp if temp == temp else 0

    q = n*c - (temp-np.sin(gamma_0)**2*np.cos(i_sat))*(d_omega_sat - d_omega_ref)-d_omega_sat*np.cos(i_sat)
    l = -r_ref*np.sin(i_sat)*np.sin(i_ref)*np.sin(omega_0)/np.sin(phi_0)*(d_omega_sat-d_omega_ref)
    l = l if l == l else 0

    def equations(p):
        m,phi = p
        return(m*np.sin(phi)-z_0,l*np.sin(phi)+q*m*np.cos(phi)-dz_0)

    #Solve simultaneous equations
    m,phi = fsolve(equations,(0,0))

    
    """
    Equations of motion in LVLH frame (from Schweighart). To be integrated!
    Inputs:
        t - time
        state - state of satellite
    Outputs:
        Derivative of the state
    """
    def J2_pert_func(t,state):
        [x,y,z] = state[:3] #Position
        [dx,dy,dz] = state[3:] #Velocity
        dX0 = dx
        dX1 = dy
        dX2 = dz

        dX3 = 2*n*c*dy + (5*c**2-2)*n**2*x - 3*n**2*J2*(R_e**2/r_ref)*(0.5 - ((3*np.sin(i_ref)**2*np.sin(k*t)**2)/2) - ((1+3*np.cos(2*i_ref))/8))
        dX4 = -2*n*c*dx - 3*n**2*J2*(R_e**2/r_ref)*np.sin(i_ref)**2*np.sin(k*t)*np.cos(k*t)
        dX5 = -q**2*z + 2*l*q*np.cos(q*t+phi)

        return np.array([dX0,dX1,dX2,dX3,dX4,dX5])

    return J2_pert_func
