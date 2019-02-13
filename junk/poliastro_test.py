import matplotlib.pyplot as plt
plt.ion()

import numpy as np
from astropy import units as u
import functools
from astropy.time import Time, TimeDelta
#from astropy.coordinates import BaseCoordinateFrame,get_body_barycentric
from poliastro.bodies import Earth,Moon, Sun
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
from poliastro.plotting import plot, OrbitPlotter3D, OrbitPlotter
#from poliastro.util import time_range
import perturbations as ptb
#from poliastro.core.perturbations import atmospheric_drag, third_body, J2_perturbation



a = 7000 * u.km #m
ecc = 0. *u.one
inc = 45 * u.deg
Om = 50 * u.deg
argp = 0 *u.deg
nu = 0 *u.deg

tofs = TimeDelta(np.linspace(0, 48.0 * u.h, num=2000))

orbit = Orbit.from_classical(Earth,a,ecc,inc,Om,argp,nu)

perturbs = [1,2]

tof = (48.0 * u.h).to(u.s).value
rr, vv = cowell(orbit, np.linspace(0, tof, 2000), ad=ptb.perturbations, index_ls = perturbs, C_D = 1,A1 = 1,m = 1,H0 = 1,rho0 = 1)

"""
perturbed_orbit = propagate(
    orbit,
    tofs,
    method=cowell,
    rtol=1e-6,
    ad=J2_perturbation,
    #index_ls = perturbs,
    J2=Earth.J2.value,
    R=Earth.R.to(u.km).value
)

rr = perturbed_orbit.data.xyz.T.to(u.km).value


r0 = np.array([-2384.46, 5729.01, 3050.46]) * u.km
v0 = np.array([-7.36138, -2.98997, 1.64354]) * u.km / u.s

orbit = Orbit.from_vectors(Earth, r0, v0)

tofs = TimeDelta(np.linspace(0, 48.0 * u.h, num=2000))

coords = propagate(
    orbit, tofs, method=cowell,
    ad=J2_perturbation, J2=Earth.J2.value, R=Earth.R.to(u.km).value
)

rr = coords.data.xyz.T.to(u.km).value
vv = coords.data.differentials["s"].d_xyz.T.to(u.km / u.s).value

# This will be easier to compute when this is solved:
# https://github.com/poliastro/poliastro/issues/257
raans = [rv2coe(k, r, v)[3] for r, v in zip(rr, vv)]













#plot(orbit)
cowell_with_J2= functools.partial(cowell, rtol=1e-6, ad=J2_perturbation, J2=Earth.J2.value, R=Earth.R.to(u.km).value)
tr = time_range(0.0, periods=1, end=tof, format='jd', scale='tdb')
rr = orbit.sample(tr,cowell_with_J2)

j_date = 2454283.0 * u.day  # setting the exact event date is important

tof = (60 * u.day).to(u.s).value

# create interpolant of 3rd body coordinates (calling in on every iteration will be just too slow)
body_r = get_body_ephem(Moon,j_date)
epoch = Time(j_date, format='jd', scale='tdb')

# multiply Moon gravity by 400 so that effect is visible :)
cowell_with_3rdbody = functools.partial(cowell, rtol=1e-6, ad=third_body,
                                        k_third=400 * Moon.k.to(u.km**3 / u.s**2).value,
                                        third_body=body_r)

tr = time_range(j_date.value, periods=1000, end=j_date.value + 60, format='jd', scale='tdb')
rr = orbit.sample(tr, method=cowell_with_3rdbody)

frame = OrbitPlotter3D()
frame.set_attractor(Earth)
#traj = BaseCoordinateFrame(rr)

frame.plot_trajectory(rr)
frame.show()
"""