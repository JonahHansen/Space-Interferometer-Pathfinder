"""
A script to plot orbits simply for illustration
Note that cartopy and Basemap both should work here.
7 Feb 2019: Version 2.0 by Jonah Hansen. Works for circular orbits.
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import astropy.units as u
from poliastro.bodies import Earth
import orbits

plt.ion()

alt = 500.0e3   #In m
R_e = Earth.R.to(u.km).value*1000  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

n_p = 1000 #Number of phases

#Orbital inclination
inc_0 = np.radians(45) #49
#Longitude of the Ascending Node
Om_0 = np.radians(-32) #-30

#Stellar vector
ra = np.radians(4) #23
dec = np.radians(20)#43

#The max distance to the other satellites in km
delta_r_max = 550e3

lines = ['r:', 'g:', 'g:']
points = ['r.', 'g.', 'g.']

#------------------------------------------------------------------------------------------
ECEF_orbit = orbits.ECEF_orbit(n_p, R_orb, delta_r_max, inc_0, Om_0, ra, dec)
LVLH_orbit = orbits.LVLH_orbit(n_p, ECEF_orbit)

ECEF_all = [ECEF_orbit.chief_pos,ECEF_orbit.deputy1_pos,ECEF_orbit.deputy2_pos]
LVLH_all = [LVLH_orbit.chief_pos,LVLH_orbit.deputy1_pos,LVLH_orbit.deputy2_pos,LVLH_orbit.s_hats]

period = ECEF_orbit.period/60 #In minutes

#Make pretty plots.
pos_ls = [] #list of positions
for im_ix, sat_phase in enumerate(np.linspace(np.pi,3.*np.pi,7)): #np.pi, 31*np.pi,450))
#for sat_phase in np.linspace(np.pi*1.45,np.pi*1.5,2):
    plt.clf()
    plt.subplot(1, 2, 1)
    map = Basemap(projection='ortho',lat_0=0,lon_0=180 - np.degrees(sat_phase*period/24/60),resolution='l')
    map.bluemarble(scale=0.4)
    plt.axis([-0.1*R_orb, 2.1*R_orb, -0.0*R_orb, 2.0*R_orb])
    lvlh_ls = []
    #Find non-vignetted parts (two vectors)
    for xyz, lvlh, point, line in zip(ECEF_all, LVLH_all, points, lines):
        visible = (xyz[:,1] > 0) | (np.sqrt(xyz[:,0]**2 + xyz[:,2]**2) > R_e)
        visible = np.concatenate(([False],visible, [False]))
        out_of_eclipse = np.where(visible[1:] & np.logical_not(visible[:-1]))[0]
        in_to_eclipse = np.where(np.logical_not(visible[1:]) & visible[:-1])[0]
        for oute, ine in zip(out_of_eclipse, in_to_eclipse):
            plt.plot(xyz[oute:ine+1,0] + R_e, xyz[oute:ine+1,2] + R_e,line)

        #Interpolate to current time.
        sat_xyz = [np.interp( (sat_phase) % (2*np.pi), ECEF_orbit.phase, xyz[:,ii]) for ii in range(3)]

        #If in foreground or more than R_earth away in (x,z) plane, plot.
        if (sat_xyz[1] > 0) | (np.sqrt(sat_xyz[0]**2 + sat_xyz[2]**2) > R_e):
            plt.plot(sat_xyz[0] + R_e, sat_xyz[2] + R_e,point)

    for lvlh in LVLH_all:
        sat_lvlh = [np.interp( (sat_phase) % (2*np.pi), ECEF_orbit.phase, lvlh[:,ii]) for ii in range(3)]
        lvlh_ls.append(sat_lvlh)

    plt.tight_layout()
    plt.subplot(336, aspect='equal')
    #plt.axes().set_aspect('equal')

    km = 1e-3

    plt.xlim(-2*delta_r_max*km,2*delta_r_max*km)
    plt.ylim(-2*delta_r_max*km,2*delta_r_max*km)

    #Star vector
    s = lvlh_ls[3]
    plt.arrow(0,0,delta_r_max*km*s[1],delta_r_max*km*s[2],width=delta_r_max*km/40,color='k')

    #List of positions
    pos_ls.append([lvlh_ls[0],lvlh_ls[1],lvlh_ls[2]])
    pos_arr = np.array(pos_ls)

    #Plot previous positions as a line
    plt.plot(pos_arr[:,0,1]*km,pos_arr[:,0,2]*km,'r--')
    plt.plot(pos_arr[:,1,1]*km,pos_arr[:,1,2]*km,'b--')
    plt.plot(pos_arr[:,2,1]*km,pos_arr[:,2,2]*km,'b--')

    #Plot the current point
    plt.plot(pos_arr[-1,0,1]*km,pos_arr[-1,0,2]*km,'ro')
    plt.plot(pos_arr[-1,1,1]*km,pos_arr[-1,1,2]*km,'bo')
    plt.plot(pos_arr[-1,2,1]*km,pos_arr[-1,2,2]*km,'bo')
    plt.title("LVLH Frame")
    plt.xlabel("Y Direction (along orbit) (km)")
    plt.ylabel("Z Direction (along chief orbital axis) (km)")

    plt.pause(.01)
