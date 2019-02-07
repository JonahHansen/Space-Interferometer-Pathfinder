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
plt.ion()

alt = 500.0e3   #In m
R_e = 6375.0e3  #In m
n_p = 1000 #Number of phases

#Orbital orientation
inc0 = np.radians(49) #Inclination
Om0 = np.radians(30) #Longitude of the accending node

#Stellar vector
ra = np.radians(0)
dec = np.radians(45)

#The distance to the other satellites in km
b = 640e3

period = 96.0 #In minutes.
lines = ['r:', 'g:', 'g:']
points = ['r.', 'g.', 'g.']
#------------------------------------------------------------------------------------------
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt
#Orbital phase space, i.e. mean longitude. n_p the number of phases
phase = np.linspace(0, 2*np.pi, n_p)

#Central spacecraft Cartesian coordinates for a circular orbit in the x,y plane.
xyzc = np.zeros( (n_p,3) )
for i in range(n_p):
    xyzc[i,0] = np.cos(phase[i]) * R_orb #x position
    xyzc[i,1] = np.sin(phase[i]) * R_orb #y position

#Cartesian coordinates for all 3 spacecraft
xyzo = np.zeros( (3,n_p,3) )

#Rotation matrix and orbital angular momentum vector for central combiner.
#Create this matrix by rotating in the inclination i, and then the longitude of
#the ascending node Omega.

R_Om = np.array([[np.cos(Om0), np.sin(Om0),0],[-np.sin(Om0), np.cos(Om0), 0],[0,0,1]]) #Rotate around z axis by Omega
R_I = np.array([[1,0,0],[0,np.cos(inc0),np.sin(inc0)], [0, -np.sin(inc0), np.cos(inc0)]]) #Rotate around x axis by i
R_0 = np.matmul(R_Om,R_I) #Full rotation matrix
o_0 = R_0[:,2] #Angular momentum is the z component (Perpendicular to x-y plane)

#Vector pointing to the star, from right ascension and declination
s_hat = [np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)]

#Now lets find the other satellites' phi and inc. Mike thought it would be cool to
#use vector arithmetic for this.
epsilon = b/R_orb
delta = epsilon
o_1 = o_0 + epsilon*np.cross(o_0,s_hat)
o_2 = o_0 - epsilon*np.cross(o_0,s_hat)
print(o_0,o_1,o_2)

#New inc and Oem
inc_1 = np.arccos(o_1[2]/np.linalg.norm(o_1))
Om_1 = np.arctan2(o_1[0],o_1[1])

inc_2 = np.arccos(o_2[2]/np.linalg.norm(o_2))
Om_2 = np.arctan2(o_2[0],o_2[1])

#Next, compute the orbital rotation matrices for these two other satellites.
Rs = [R_0]
poffsets = [0,-delta, delta]
print(Om0, inc0)
for Om, i in zip([Om_1, Om_2], [inc_1, inc_2]):
    print(Om, i)
    R_Om = np.array([[np.cos(Om), np.sin(Om),0],[-np.sin(Om), np.cos(Om), 0],[0,0,1]]) #Rotate around z axis by Omega
    R_I = np.array([[1,0,0],[0,np.cos(i),np.sin(i)], [0, -np.sin(i), np.cos(i)]]) #Rotate around new x axis by i
    Rs.append(np.matmul(R_Om,R_I))

#Use the Rotation matrices to find the orbital positions.
for ix, R in enumerate(Rs):
    for iy in range(n_p):
        xyzo[ix,iy] = np.dot(R, xyzc[iy])

#Lets compute dot products for the full orbit
for ix in range(n_p):
    sep1 = np.abs(xyzo[1,ix] - xyzo[0,ix])
    sep2 = np.abs(xyzo[2,ix] - xyzo[0,ix])
    total_sep = np.abs(xyzo[2,ix] - xyzo[1,ix])
    #print(np.linalg.norm(sep1+sep2),np.linalg.norm(total_sep))
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
    plt.pause(.1)
