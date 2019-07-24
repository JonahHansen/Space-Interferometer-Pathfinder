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
import astropy.constants as const
import modules.orbits as orbits

plt.ion()

alt = 500.0e3   #In m
R_e = const.R_earth.value  #In m

#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

n_p = 1000 #Number of phases

#Orbital inclination
inc_0 = np.radians(0)
#Longitude of the Ascending Node
Om_0 = np.radians(0)

#Stellar vector
ra = np.radians(0)
dec = np.radians(45)

#The max distance to the other satellites in m
delta_r_max = 1000e3

lines = ['r:', 'g:', 'g:']
points = ['r.', 'g.', 'g.']

#------------------------------------------------------------------------------------------
#Calculate reference orbit, in the geocentric (ECI) frame (See Orbit module)
ref = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

num_times = 1000
times = np.linspace(0,ref.period,num_times)

c_pos = np.zeros((num_times,3))
dep1_pos = np.zeros((num_times,3))
dep2_pos = np.zeros((num_times,3))
LVLH_pos0 = np.zeros((num_times,3))
LVLH_pos1 = np.zeros((num_times,3))
LVLH_pos2 = np.zeros((num_times,3))
Base_pos0 = np.zeros((num_times,3))
Base_pos1 = np.zeros((num_times,3))
Base_pos2 = np.zeros((num_times,3))
s_hats = np.zeros((num_times,3))

i = 0
for t in times:
    pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(t)
    chief = orbits.init_chief(ref,t)
    c_pos[i] = chief.pos
    dep1 = orbits.init_deputy(ref,t,1)
    dep2 = orbits.init_deputy(ref,t,2)
    dep1_pos[i] = dep1.pos
    dep2_pos[i] = dep2.pos
    LVLH_pos0[i] = chief.to_LVLH(pos_ref,vel_ref,LVLH).pos
    LVLH_pos1[i] = dep1.to_LVLH(pos_ref,vel_ref,LVLH).pos
    LVLH_pos2[i] = dep2.to_LVLH(pos_ref,vel_ref,LVLH).pos
    Base_pos0[i] = chief.to_LVLH(pos_ref,vel_ref,LVLH).to_Baseline(LVLH,Base).pos
    Base_pos1[i] = dep1.to_LVLH(pos_ref,vel_ref,LVLH).to_Baseline(LVLH,Base).pos
    Base_pos2[i] = dep2.to_LVLH(pos_ref,vel_ref,LVLH).to_Baseline(LVLH,Base).pos
    s_hats[i] = np.dot(LVLH,ref.s_hat)
    i += 1

#All ECI positions
ECI_all = [c_pos,dep1_pos,dep2_pos]
#All LVLH positions, plus stellar vector
LVLH_all = [LVLH_pos0,LVLH_pos1,LVLH_pos2,s_hats]
#All Baseline positions
Base_all = [Base_pos0,Base_pos1,Base_pos2]

period = ref.period/60 #In minutes

plt.figure(1,figsize=(8,7.5))

#Make pretty plots.
lvlh_pos_ls = [] #list of lvlh positions
base_pos_ls = [] #list of baseline positions
for im_ix, sat_phase in enumerate(np.linspace(1.*np.pi,7*np.pi,300)): #np.pi, 31*np.pi,450))
#for sat_phase in np.linspace(np.pi*1.45,np.pi*1.5,2):
    plt.clf()
    plt.subplot(1, 2, 1)
    map = Basemap(projection='ortho',lat_0=0,lon_0=180 - np.degrees(sat_phase*period/24/60),resolution='l')
    map.bluemarble(scale=0.4)
    plt.axis([-0.1*R_orb, 2.1*R_orb, -0.0*R_orb, 2.0*R_orb])
    lvlh_ls = []
    base_ls = []
    #Find non-vignetted parts (two vectors)
    for xyz, point, line in zip(ECI_all, points, lines):
        visible = (-xyz[:,1] > 0) | (np.sqrt(xyz[:,0]**2 + xyz[:,2]**2) > R_e)
        visible = np.concatenate(([False],visible, [False]))
        out_of_eclipse = np.where(visible[1:] & np.logical_not(visible[:-1]))[0]
        in_to_eclipse = np.where(np.logical_not(visible[1:]) & visible[:-1])[0]
        for oute, ine in zip(out_of_eclipse, in_to_eclipse):
            plt.plot(xyz[oute:ine+1,0] + R_e, xyz[oute:ine+1,2] + R_e,line)

        #Interpolate to current time.
        sat_xyz = [np.interp( (sat_phase) % (2*np.pi), ref.ang_vel*times, xyz[:,ii]) for ii in range(3)]

        #If in foreground or more than R_earth away in (x,z) plane, plot.
        if (-sat_xyz[1] > 0) | (np.sqrt(sat_xyz[0]**2 + sat_xyz[2]**2) > R_e):
            plt.plot(sat_xyz[0] + R_e, sat_xyz[2] + R_e,point)

    #Interpolate LVLH orbit, to make LVLH plot
    for lvlh in LVLH_all:
        sat_lvlh = [np.interp( (sat_phase) % (2*np.pi), ref.ang_vel*times, lvlh[:,ii]) for ii in range(3)]
        lvlh_ls.append(sat_lvlh)

    #Interpolate LVLH orbit, to make LVLH plot
    for base in Base_all:
        sat_base = [np.interp( (sat_phase) % (2*np.pi), ref.ang_vel*times, base[:,ii]) for ii in range(3)]
        base_ls.append(sat_base)

    plt.tight_layout()
    plt.subplot(233, aspect='equal')
    #plt.axes().set_aspect('equal')

    km = 1e-3

    plt.xlim(-2*delta_r_max*km,2*delta_r_max*km)
    plt.ylim(-2*delta_r_max*km,2*delta_r_max*km)

    #Star vector
    s =  lvlh_ls[3]

    #List of positions
    lvlh_pos_ls.append([lvlh_ls[0],lvlh_ls[1],lvlh_ls[2]])
    pos_arr = np.array(lvlh_pos_ls)

    #Plot previous positions as a line
    plt.plot(pos_arr[:,0,1]*km,pos_arr[:,0,2]*km,'r--')
    plt.plot(pos_arr[:,1,1]*km,pos_arr[:,1,2]*km,'b--')
    plt.plot(pos_arr[:,2,1]*km,pos_arr[:,2,2]*km,'b--')

    #Plot the current point
    plt.plot(pos_arr[-1,0,1]*km,pos_arr[-1,0,2]*km,'ro')
    plt.plot(pos_arr[-1,1,1]*km,pos_arr[-1,1,2]*km,'bo')
    plt.plot(pos_arr[-1,2,1]*km,pos_arr[-1,2,2]*km,'bo')
    plt.title("LVLH Frame")
    plt.xlabel("v (chief velocity axis) (km)")
    plt.ylabel("h (chief OAM axis) (km)")

    plt.arrow(0,0,delta_r_max*km*s[1],delta_r_max*km*s[2],width=delta_r_max*km/40,color='k')

    plt.subplot(236, aspect='equal')
    #plt.axes().set_aspect('equal')

    km = 1e-3

    plt.xlim(-2*delta_r_max*km,2*delta_r_max*km)
    plt.ylim(-2*delta_r_max*km,2*delta_r_max*km)

    #Star vector
    s = np.array([0,0,1])

    #List of positions
    base_pos_ls.append([base_ls[0],base_ls[1],base_ls[2]])
    pos_arr = np.array(base_pos_ls)

    #Plot previous positions as a line
    plt.plot(pos_arr[:,0,0]*km,pos_arr[:,0,2]*km,'r--')
    plt.plot(pos_arr[:,1,0]*km,pos_arr[:,1,2]*km,'b--')
    plt.plot(pos_arr[:,2,0]*km,pos_arr[:,2,2]*km,'b--')

    #Plot the current point
    plt.plot(pos_arr[-1,0,0]*km,pos_arr[-1,0,2]*km,'ro')
    plt.plot(pos_arr[-1,1,0]*km,pos_arr[-1,1,2]*km,'bo')
    plt.plot(pos_arr[-1,2,0]*km,pos_arr[-1,2,2]*km,'bo')
    plt.title("Baseline Frame")
    plt.xlabel("b (baseline axis) (km)")
    plt.ylabel("s (star vector axis) (km)")

    plt.arrow(0,0,delta_r_max*km*s[1],delta_r_max*km*s[2],width=delta_r_max*km/40,color='k')

    plt.savefig("pngs/orb{:03d}.png".format(im_ix))
    plt.pause(.01)
