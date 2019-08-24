import numpy as np
import astropy.constants as const
from scipy.optimize import fsolve

""" Implementation of the solutions to Schweghart's equations of motion """

""" equations_creation: takes in the reference orbit and produces a function """
""" that takes in the initial conditions for the integrated equations """

""" set_init_conditions: takes in initial conditions and returns a Function """
""" that takes in a time and produces the satellite's state relative to the """
""" reference orbit at that time."""

def propagate_spacecraft(t0,state0,t,ref,HCW=False):

    r_ref = ref.R_orb #Reference orbit radius

    #Schweighart constants (defined in notebook)
    n = ref.ang_vel
    if HCW:
        c = 1
        k = n
        J2 = 0
    else:
        c = ref.Sch_c
        k = ref.Sch_k
        J2 = 0.00108263

    Re = const.R_earth.value
    i_ref = ref.inc_0 -(3*n*J2*Re**2)/(2*k*r_ref**2)*np.cos(ref.inc_0)*np.sin(ref.inc_0)#Inclination of the reference orbit (and chief)

    #Self defined constants for the equations of motion
    a = 2*n*c
    b = (5*c**2-2)*n**2
    g = 3*n**2*J2*Re**2/r_ref*np.sin(i_ref)**2
    d = np.sqrt(a**2-b)
    e = (d**2)-4*k**2

    print(a,b,g,d,e)

    x0,y0,z0,dx0,dy0,dz0 = state0 #Initial conditions

    #Same constants as defined in Schweighart
    i_sat = dz0/(k*r_ref)+i_ref

    if i_ref == 0:
        omega_0 = 0
    else:
        omega_0 = z0/(r_ref*np.sin(i_ref))

    if (omega_0 and i_ref) != 0:
        gamma_0 = np.arctan(np.sin(omega_0)/(1/np.tan(i_ref)*np.sin(i_sat)-np.cos(i_sat)*np.cos(omega_0)))
    else:
        gamma_0 = 0

    phi_0 = np.arccos(np.cos(i_sat)*np.cos(i_ref)+np.sin(i_sat)*np.sin(i_ref)*np.cos(omega_0))

    d_omega_sat = -3*n*J2*Re**2/(2*r_ref**2)*np.cos(i_sat)

    d_omega_ref = -3*n*J2*Re**2/(2*r_ref**2)*np.cos(i_ref)

    temp = np.cos(gamma_0)*np.sin(gamma_0)*1/np.tan(omega_0)
    temp = temp if temp == temp else 0

    q = n*c - (temp-np.sin(gamma_0)**2*np.cos(i_sat))*(d_omega_sat - d_omega_ref)-d_omega_sat*np.cos(i_sat)

    l = -r_ref*np.sin(i_sat)*np.sin(i_ref)*np.sin(omega_0)/np.sin(phi_0)*(d_omega_sat-d_omega_ref)
    l = l if l == l else 0

    def equations(p):
        m,phi = p
        return(m*np.sin(phi)-z0,l*np.sin(phi)+q*m*np.cos(phi)-dz0)

    #Solve simultaneous equations
    m,phi = fsolve(equations,(0,0))

    ka1 = b/(d**2)*(a*x0 + dy0 - g/(4*k)*np.cos(2*k*t0))
    ka2 = y0 - a/(d**2)*dx0 + g*(b-3*a*k)/(8*d**2*k**2)*np.sin(2*k*t0)
    ka3 = -(b/(d**2)*x0 + a/d**2*dy0 + g/(4*d**2*e)*(4*a*k - 3*d**2)*np.cos(2*k*t0))
    ka4 = 1/(2*d*e)*(2*e*dx0 + g*(a-3*k)*np.sin(2*k*t0))
    ka5 = g/(4*e*k)*(a-3*k)
    ka6 = g/(8*b*k**2)*(3*a*k - b - 4*k**2)

    print(ka4)

    #import pdb; pdb.set_trace()
    #Actual equations
    x = a/b*ka1 + ka3*np.cos(d*(t-t0)) + ka4*np.sin(d*(t-t0)) + ka5*np.cos(2*k*t)
    y = ka2 - ka1*(t-t0) + a/d*ka4*np.cos(d*(t-t0)) -a/d*ka3*np.sin(d*(t-t0)) + ka6*np.sin(2*k*t)
    z = z0*np.cos(q*(t-t0)) + l*(t-t0)*np.sin(q*t+phi) + 1/q*np.sin(q*(t-t0))*(dz0 - l*np.sin(q*t0+phi))

    dx = - d*ka3*np.sin(d*(t-t0)) + d*ka4*np.cos(d*(t-t0)) - 2*k*ka5*np.sin(2*k*t)
    dy = - ka1 - a*ka4*np.sin(d*(t-t0)) - a*ka3*np.cos(d*(t-t0)) + 2*k*ka6*np.cos(2*k*t)
    dz = (l*q*(t-t0)*np.cos(q*t+phi) - q*z0*np.sin(q*(t-t0)) +
         l*np.sin(q*t+phi) + np.cos(q*(t-t0))*(dz0-l*np.sin(q*t0+phi)))

    return np.array([x,y,z,dx,dy,dz])
