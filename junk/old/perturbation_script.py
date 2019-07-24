
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from astropy import units as u
from astropy.time import Time
from perturbation_integrate import perturb_orbit, none_dX_dt
import orbits

plt.ion()

alt = 1000e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(34) #49
#Longitude of the Ascending Node
Om_0 = np.radians(23) #-30

#Stellar vector
ra = np.radians(12) #23
dec = np.radians(-75)#43

#The max distance to the other satellites in m
delta_r_max = 0.3*1e3

n_p = 1000 #number of phases
#------------------------------------------------------------------------------------------
#Calculate orbit, in the geocentric (ECI) frame
ECI = orbits.ECI_orbit(n_p, R_orb, delta_r_max, inc_0, Om_0, ra, dec)

#Convert orbit into the LVLH frame
LVLH = orbits.LVLH_orbit(n_p, R_orb, ECI)

#Peturb the orbit according to the HCW equations
#NOTE SHOULD GIVE SAME AS LVLH ORBIT!!!! BUT DOESNT!!!! SOMETHING BROKEN!!!
pert_LVLH = perturb_orbit(LVLH,none_dX_dt)
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

#Plot ECI Orbit
plt.figure(1)
ax1 = plt.axes(projection='3d')
ax1.set_aspect('equal')
ax1.plot3D(ECI.chief_pos[:,0],ECI.chief_pos[:,1],ECI.chief_pos[:,2],'b-')
ax1.plot3D(ECI.deputy1_pos[:,0],ECI.deputy1_pos[:,1],ECI.deputy1_pos[:,2],'r-')
ax1.plot3D(ECI.deputy2_pos[:,0],ECI.deputy2_pos[:,1],ECI.deputy2_pos[:,2],'g-')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('z (m)')
ax1.set_title('Orbit in ECI frame')
set_axes_equal(ax1)

#Plot LVLH (solid) and perturbed LVLH (dashed) orbits
plt.figure(2)
ax2 = plt.axes(projection='3d')
ax2.set_aspect('equal')
ax2.plot3D(LVLH.chief_pos[:,0],LVLH.chief_pos[:,1],LVLH.chief_pos[:,2],'b-')
ax2.plot3D(LVLH.deputy1_pos[:,0],LVLH.deputy1_pos[:,1],LVLH.deputy1_pos[:,2],'r-')
ax2.plot3D(LVLH.deputy2_pos[:,0],LVLH.deputy2_pos[:,1],LVLH.deputy2_pos[:,2],'g-')
ax2.plot3D(pert_LVLH.chief_pos[:,0],pert_LVLH.chief_pos[:,1],pert_LVLH.chief_pos[:,2],'b--')
ax2.plot3D(pert_LVLH.deputy1_pos[:,0],pert_LVLH.deputy1_pos[:,1],pert_LVLH.deputy1_pos[:,2],'r--')
ax2.plot3D(pert_LVLH.deputy2_pos[:,0],pert_LVLH.deputy2_pos[:,1],pert_LVLH.deputy2_pos[:,2],'g--')
ax2.set_xlabel('r (m)')
ax2.set_ylabel('v (m)')
ax2.set_zlabel('h (m)')
ax2.set_title('Orbit in LVLH frame')
set_axes_equal(ax2)