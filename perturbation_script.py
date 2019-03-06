
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from astropy import units as u
from astropy.time import Time
from perturbation_integrate import perturb_orbit, none_dX_dt, J2_dX_dt
import orbits

plt.ion()

alt = 1000e3 #In m

#Orbital inclination
inc_0 = np.radians(20) #49
#Longitude of the Ascending Node
Om_0 = np.radians(34) #-30

#Stellar vector
ra = np.radians(52) #23
dec = np.radians(45)#43

#The max distance to the other satellites in km
delta_r_max = 0.3*1e3

#Perturbations (see module)
j_date = 2454283.0 * u.day #Epoch

n_p = 1000
#------------------------------------------------------------------------------------------
#Orbital radius the sum of earth radius and altitude
R_e = const.R_earth.value  #In m
R_orb = R_e + alt

ECEF_orbit = orbits.ECEF_orbit(n_p, R_orb, delta_r_max, inc_0, Om_0, ra, dec)
LVLH_orbit = orbits.LVLH_orbit(n_p, R_orb, ECEF_orbit)

perturbed_LVLH = perturb_orbit(LVLH_orbit,none_dX_dt)
# ---------------------------------------------------------------------- #
### PLOTTING STUFF ###

### Functions to set 3D axis aspect ratio as equal
def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


#plt.show()
