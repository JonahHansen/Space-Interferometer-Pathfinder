"""
A script to plot orbits simply for illustration

Note that cartopy and Basemap both should work here.

1 Feb 2019: Initial version by Mike Ireland. Still has a bug.
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const
import astropy.units as u
from mpl_toolkits.basemap import Basemap
import quaternions as qt
plt.ion()

alt = 500.0e3   #In m
R_e = 6375.0e3  #In m
n_p = 1000 #Number of phases

#Orbital inclination
inc_0 = np.radians(49)
Om_0 = np.radians(-30)

#Stellar vector
ra = np.radians(23)
dec = np.radians(43)

#The distance to the other satellites in km
b = 640e3

period = 96.0 #In minutes.
lines = ['r:', 'g:', 'g:']
points = ['r.', 'g.', 'g.']

xaxis = np.array([1,0,0])
yaxis = np.array([0,1,0])
zaxis = np.array([0,0,1])
#------------------------------------------------------------------------------------------
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt
#Orbital phase, i.e. mean longitude. n_p the number of phases
phase = np.linspace(0, 2*np.pi, n_p)

#Central spacecraft Cartesian coordinates for a circular orbit in the x,y plane.
xyzc = np.zeros( (n_p,3) )
for i in range(n_p):
    xyzc[i,0] = np.cos(phase[i]) * R_orb
    xyzc[i,1] = np.sin(-phase[i]) * R_orb

#Cartesian coordinates for all 3 spacecraft
xyzo = np.zeros( (3,n_p,3) )

#Quaternion rotation, using Mike's "No phase difference" rotation!

q_Om = qt.to_q(zaxis,Om_0)
q_inc = qt.to_q(yaxis,inc_0)
q_0 = qt.comb_rot(qt.comb_rot(qt.conjugate(q_Om),q_inc),q_Om)

o_0 = qt.rotate(zaxis,q_0) #Angular momentum vector of chief satellite

#Vector pointing to the star, from right ascension and declination
s_hat = [np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)]

#Now lets find the other satellites' phi and inc. Mike thought it would be cool to
#use vector arithmetic for this.
epsilon = b/R_orb
delta = epsilon #* np.dot(o_0, s_hat)
o_1 = o_0 + epsilon*np.cross(o_0,s_hat)
o_2 = o_0 - epsilon*np.cross(o_0,s_hat)
o_1 /= np.linalg.norm(o_1)
o_2 /= np.linalg.norm(o_2)
inc_1 = np.arccos(o_1[2])
Om_1= np.arctan2(o_1[1], o_1[0])
inc_2 = np.arccos(o_2[2])
Om_2= np.arctan2(o_2[1], o_2[0])

#Next, compute the orbital rotation quaternions for these two other satellites.
qs = [q_0]
poffsets = [0,-delta, delta]
print(Om_0, inc_0)
for Om, inc in zip([Om_1, Om_2], [inc_1, inc_2]):
    print(Om, inc)
    q_Om = qt.to_q(zaxis,Om)
    q_inc = qt.to_q(yaxis,inc)
    q_0 = qt.comb_rot(qt.comb_rot(qt.conjugate(q_Om),q_inc),q_Om)
    qs.append(q_0)

#Use the quaternions to find the orbital positions.
for ix, q in enumerate(qs):
    xyzo[ix] = qt.rotate_points(xyzc,q)

#Lets compute dot products for the full orbit

for ix in range(n_p):
    sep1 = xyzo[1,ix] - xyzo[0,ix]
    sep2 = xyzo[2,ix] - xyzo[0,ix]
    sep1 /= np.linalg.norm(sep1)
    sep2 /= np.linalg.norm(sep2)
    print(sep1,sep2,s_hat)
    print("Dot products with star vector: {:6.1f} {:6.1f}".format(np.dot(s_hat, sep1), np.dot(s_hat, sep2)))


#Make pretty plots.
for im_ix, sat_phase in enumerate(np.linspace(np.pi,2.*np.pi,6)): #np.pi, 31*np.pi,450))
#for sat_phase in np.linspace(np.pi*1.45,np.pi*1.5,2):
    plt.clf()
    map = Basemap(projection='ortho',lat_0=0,lon_0=180 - np.degrees(sat_phase*period/24/60),resolution='l')
    map.bluemarble(scale=0.4)
    plt.axis([-0.1*R_orb, 2.1*R_orb, -0.0*R_orb, 2.0*R_orb])
    #Find non-vignetted parts (two vectors)
    for xyz, poffset, point, line in zip(xyzo, poffsets, points, lines):
        visible = (xyz[:,1] > 0) | (np.sqrt(xyz[:,0]**2 + xyz[:,2]**2) > R_e)
        visible = np.concatenate(([False],visible, [False]))
        out_of_eclipse = np.where(visible[1:] & np.logical_not(visible[:-1]))[0]
        in_to_eclipse = np.where(np.logical_not(visible[1:]) & visible[:-1])[0]
        for oute, ine in zip(out_of_eclipse, in_to_eclipse):
            plt.plot(xyz[oute:ine+1,0] + R_e, xyz[oute:ine+1,2] + R_e,line)

        #Interpolate to current time.
        sat_xyz = [np.interp( (sat_phase-poffset) % (2*np.pi), phase, xyz[:,ii]) for ii in range(3)]

        #If in foreground or more than R_earth away in (x,z) plane, plot.
        if (sat_xyz[1] > 0) | (np.sqrt(sat_xyz[0]**2 + sat_xyz[2]**2) > R_e):
            plt.plot(sat_xyz[0] + R_e, sat_xyz[2] + R_e,point)
    plt.tight_layout()
    #plt.savefig("pngs/orb{:03d}.png".format(im_ix))