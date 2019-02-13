""" Perturbations module """

import numpy as np
from astropy import units as unit
from poliastro.core.perturbations import atmospheric_drag, third_body, J2_perturbation, radiation_pressure
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

def perturbations(t0,u,k,index_ls,**kwargs):

    final_accel = 0 #Starting acceleration
    R_E=Earth.R.to(unit.km).value #Earth's Radius

    ### No perturbation = 0 ###
    if 0 in index_ls:

        return np.zeros(3)

    ### J2 = 1 ###
    if 1 in index_ls:

        J2_acc = J2_perturbation(t0,u,k,Earth.J2.value,R_E)
        final_accel += J2_acc


    ### Drag = 2 ###
    if 2 in index_ls:

        #C_D (float) – dimensionless drag coefficient ()
        #A (float) – frontal area of the spacecraft (km^2)
        #m (float) – mass of the spacecraft (kg)
        #H0 (float) – atmospheric scale height (km)
        #rho0 (float) – the exponent density pre-factor (kg / m^3)

        drag = atmospheric_drag(t0,u,k,R_E,kwargs["C_D"],kwargs["A1"],kwargs["m"],kwargs["H0"],kwargs["rho0"])
        final_accel += drag


    ### Third Body = 3 ###
    if 3 in index_ls:

        #k (float) – gravitational constant of third body (km^3/s^2)
        #third_body (a callable object returning the position of 3rd body) – third body that causes the perturbation

        t_body = third_body(t0,u,k,kwargs["k_third"],kwargs["third_body"])
        final_accel += t_body


    ### Rad_pressure = 4 ###
    if 4 in index_ls:

        #C_R (float) – dimensionless radiation pressure coefficient, 1 < C_R < 2 ()
        #A (float) – effective spacecraft area (km^2)
        #m (float) – mass of the spacecraft (kg)
        #Wdivc_s (float) – total star emitted power divided by the speed of light (W * s / km)
        #star (a callable object returning the position of star in attractor frame) – star position

        rad_pressure = (t0,u,k,R_E,kwargs["C_R"],kwargs["A2"],kwargs["m"],kwargs["Wdivc_s"],kwargs["star"])
        final_accel += rad_pressure


    return final_accel

#Take two consecutive positions, and the period, to calculate the velocity vectors
# and then calculate the poliastro orbit
def from_pos_to_orbit(pos1,pos2,n_p,period):
    ave_pos = 0.5*(pos1+pos2) #Average position vector between the two
    sep = pos2-pos1 #Separation vector

    #Velocity = dr/dt = dr/(T/num_phases)
    vel = sep*n_p/period

    #Make orbit
    orbit = Orbit.from_vectors(Earth, ave_pos*unit.km, vel*unit.km / unit.s)
    return orbit
