import matplotlib.pyplot as plt
plt.ion()

import numpy as np
from astropy import units as u
import functools
from astropy.time import Time, TimeDelta
#from astropy.coordinates import BaseCoordinateFrame,get_body_barycentric
from poliastro.bodies import Earth,Moon, Sun
from poliastro.twobody import Orbit
from poliastro.ephem import build_ephem_interpolant
from astropy.coordinates import solar_system_ephemeris
from poliastro.twobody.propagation import cowell
from poliastro.plotting import plot, OrbitPlotter3D, OrbitPlotter
#from poliastro.util import time_range
#from poliastro.core.perturbations import atmospheric_drag, third_body, J2_perturbation


# database keeping positions of bodies in Solar system over time
solar_system_ephemeris.set('de432s')

j_date = 2454283.0 * u.day  # setting the exact event date is important

tof = (60 * u.day).to(u.s).value
epoch = Time(j_date, format='jd', scale='tdb')

# create interpolant of 3rd body coordinates (calling in on every iteration will be just too slow)
body_r = build_ephem_interpolant(Moon, 28 * u.day, (j_date, j_date + 60 * u.day), rtol=1e-2)

a = 7000 * u.km #m
ecc = 0. *u.one
inc = 45 * u.deg
Om = 50 * u.deg
argp = 0 *u.deg
nu = 0 *u.deg

tofs = TimeDelta(np.linspace(0, 48.0 * u.h, num=2000))

orbit = Orbit.from_classical(Earth,a,ecc,inc,Om,argp,nu,epoch=epoch)
