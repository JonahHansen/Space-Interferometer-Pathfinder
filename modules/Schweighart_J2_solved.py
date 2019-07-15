import numpy as np
import astropy.constants as const
from scipy.optimize import fsolve

""" Implementation of the solutions to Schweghart's equations of motion """

""" equations_creation: takes in the reference orbit and produces a function """
""" that takes in the initial conditions for the integrated equations """

""" set_init_conditions: takes in initial conditions and returns a Function """
""" that takes in a time and produces the satellite's state relative to the """
""" reference orbit at that time."""

def equations_creation(ref):

    r_ref = ref.R_orb #Reference orbit radius
    i_ref = ref.inc_0 #Inclination orbit radius

    #Schweighart constants (defined in notebook)
    c = ref.Sch_c
    w = ref.ang_vel
    k = ref.Sch_k

    J2 = 0.00108263
    Re = const.R_earth.value

    #Self defined constants for the equations of motion
    a = 2*w*c
    b = (5*c**2-2)*w**2
    d = 3*w**2*J2*Re**2/r_ref*np.sin(i_ref)**2
    g = 3*w**2*J2*Re**2/r_ref*(0.5 - (1+3*np.cos(2*i_ref))/8)
    al = np.sqrt(a**2-b)
    be = (al**2)-4*k**2

    la0 = -1/(4*al**2*be*k)
    ka0 = -1/(8*al**2*be*k**2)

    def set_init_conditions(t0,state):
        x0,y0,z0,dx0,dy0,dz0 = state #Initial conditions

        #Same constants as defined in Schweighart
        i_sat = dz0/(k*r_ref)+i_ref

        if i_ref == 0:
            omega_0 = 0
        else:
            omega_0 = z0/(r_ref*np.sin(i_ref))

        if (omega_0 and i_ref) != 0:
            gamma_0 = np.pi/2 - np.arctan((1/np.tan(i_ref)*np.sin(i_sat)-np.cos(i_sat)*np.cos(omega_0))/np.sin(omega_0))
        else:
            gamma_0 = 0

        phi_0 = np.arccos(np.cos(i_sat)*np.cos(i_ref)+np.sin(i_sat)*np.sin(i_ref)*np.cos(omega_0))
        d_omega_sat = -3*w*J2*Re**2/(2*r_ref**2)*np.cos(i_sat)
        d_omega_ref = -3*w*J2*Re**2/(2*r_ref**2)*np.cos(i_ref)

        temp = np.cos(gamma_0)*np.sin(gamma_0)*1/np.tan(omega_0)
        temp = temp if temp == temp else 0

        q = w*c - (temp-np.sin(gamma_0)**2*np.cos(i_sat))*(d_omega_sat - d_omega_ref)-d_omega_sat*np.cos(i_sat)
        l = -r_ref*np.sin(i_sat)*np.sin(i_ref)*np.sin(omega_0)/np.sin(phi_0)*(d_omega_sat-d_omega_ref)
        l = l if l == l else 0

        def equations(p):
            m,phi = p
            return(m*np.sin(phi)-z0,l*np.sin(phi)+q*m*np.cos(phi)-dz0)

        #Solve simultaneous equations
        m,phi = fsolve(equations,(1,1))

        #Constants of the solved equations
        la1 = -(be*k*(3*d-4*g+4*a*(a*x0+dy0))-a*d*be*np.cos(2*k*t0))
        la2 = -al**2*d*(a-3*k)
        la3 = k*(be*(3*d-4*g+4*(b*x0+a*dy0))+d*(-3*a**2+3*b+4*a*k)*np.cos(2*k*t0))
        la4 = -2*al*k*(2*be*dx0+d*(a-3*k)*np.sin(2*k*t0))

        ka1 = (2*k*(be*k*(-4*a**2*y0-a*(3*d-4*g+4*b*x0)*t0+4*a*dx0+4*b*(y0-dy0*t0)) +
              b*d*be*t0*np.cos(2*k*t0)) - be*d*(b-3*a*k)*np.sin(2*k*t0))
        ka2 = 2*k*(be*k*(a*(3*d-4*g+4*b*x0)+4*b*dy0)-b*d*be*np.cos(2*k*t0))
        ka3 = al**2*d*(b-3*a*k+4*k**2)
        ka4 = -4*a*k**2*(2*be*dx0 + d*(a-3*k)*np.sin(2*k*t0))
        ka5 = 2*a*k**2/al*(-be*(3*d-4*g+4*b*x0+4*a*dy0)+d*(3*a**2-3*b-4*a*k)*np.cos(2*k*t0))

        #Actual equations
        def equation(t):
            x = la0*(la1 + la2*np.cos(2*k*t) + la3*np.cos(al*(t-t0)) + la4*np.sin(al*(t-t0)))
            y = ka0*(ka1 + ka2*t + ka3*np.sin(2*k*t) + ka4*np.cos(al*(t-t0)) + ka5*np.sin(al*(t-t0)))
            z = z0*np.cos(q*(t-t0)) + l*(t-t0)*np.sin(q*t+phi) + 1/q*np.sin(q*(t-t0))*(dz0 - l*np.sin(q*t0+phi))

            dx = la0*(-2*k*la2*np.sin(2*k*t) - al*la3*np.sin(al*(t-t0)) + al*la4*np.cos(al*(t-t0)))
            dy = ka0*(ka2 + 2*k*ka3*np.cos(2*k*t) - al*ka4*np.sin(al*(t-t0)) + al*ka5*np.cos(al*(t-t0)))
            dz = (l*q*(t-t0)*np.cos(q*t+phi) - q*z0*np.sin(q*(t-t0)) +
                 l*np.sin(q*t+phi) + np.cos(q*(t-t0))*(dz0-l*np.sin(q*t0+phi)))

            return np.array([x,y,z,dx,dy,dz])
        return equation
    return set_init_conditions
