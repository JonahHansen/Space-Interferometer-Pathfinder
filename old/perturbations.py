""" Perturbations module """

import numpy as np
from astropy import units as unit
import astropy.constants as const
from poliastro.core.perturbations import atmospheric_drag, third_body, J2_perturbation, radiation_pressure
from poliastro.bodies import Earth, Moon, Sun
from poliastro.twobody import Orbit
from poliastro.ephem import build_ephem_interpolant
from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time

#Perturbation partial function to combine perturbations
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


    ### Third Body: MOON = 3 ###
    if 3 in index_ls:

        #third_body (a callable object returning the position of 3rd body) – third body that causes the perturbation

        t_body = third_body(t0,u,k,Moon.k.to(unit.km**3 / unit.s**2).value,kwargs["moon"])
        final_accel += t_body


    ### Third Body: SUN = 4 ###
    if 4 in index_ls:

        #third_body (a callable object returning the position of 3rd body) – third body that causes the perturbation

        t_body = third_body(t0,u,k,Sun.k.to(unit.km**3 / unit.s**2).value,kwargs["sun"])
        final_accel += t_body

    ### Rad_pressure of sun = 5 ###
    if 5 in index_ls:

        #C_R (float) – dimensionless radiation pressure coefficient, 1 < C_R < 2 ()
        #A (float) – effective spacecraft area (km^2)
        #m (float) – mass of the spacecraft (kg)
        #W (float) – total star emitted power (W)

        #4 Pi in there due to incorrect module
        Wdivc_s = const.L_sun/(4*np.pi*const.c.to('km/s'))

        rad_pressure = (t0,u,k,R_E,kwargs["C_R"],kwargs["A2"],kwargs["m"],Wdivc_s,kwargs["sun"])
        final_accel += rad_pressure


    return final_accel

#Build moon ephemeris for perturbation #3
def moon_ephem(tof,j_date):
    solar_system_ephemeris.set('de432s')
    body_r = build_ephem_interpolant(Moon, 28 * unit.day, (j_date, j_date + tof* unit.day), rtol=1e-2)
    return body_r

#Build sun ephemeris for perturbation #4
def sun_ephem(tof,j_date):
    solar_system_ephemeris.set('de432s')
    body_r = build_ephem_interpolant(Sun, 365 * unit.day, (j_date, j_date + tof* unit.day), rtol=1e-2)
    return body_r
