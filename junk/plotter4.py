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

#Orbital inclination
inc0 = np.radians(49)
phi0 = np.radians(30)

#Stellar vector
ra = np.radians(45)
dec = np.radians(0)

#The distance to the other satellites in km
b = 640e3

period = 96.0 #In minutes.
lines = ['r:', 'g:', 'g:']
points = ['r.', 'g.', 'g.']
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

#Rotation matrix and orbital angular momentum vector for central combiner.
#Create this matrix by rotating in azimuthal angle phi (longitude of the ascending (?)
#node) then rotating in inclination,
#called both th and inc below. Probably use conventional terms i and Omega!

R_phi = np.array([[np.cos(phi0), -np.sin(phi0),0],[np.sin(phi0), np.cos(phi0), 0],[0,0,1]])
R_th = np.array([[np.cos(inc0),0,np.sin(inc0)], [0, 1, 0], [-np.sin(inc0), 0, np.cos(inc0)]])
R_0 = R_phi.dot(R_th).dot(np.linalg.inv(R_phi))
o_0 = R_0[:,2]

#Vector pointing to the star, from right ascension and declination
s_hat = [np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)]

pt1_0 = np.dot(R_0, xyzc[0])
perp_ln_1 = np.cross(s_hat,pt1_0/np.linalg.norm(pt1_0))

pt1_1 = pt1_0 + b*perp_ln_1
pt1_2 = pt1_0 - b*perp_ln_1

deputy_rad = np.linalg.norm(pt1_1)

n_hat1 = pt1_1/deputy_rad

u_hat1 = np.cross(n_hat1,np.array([1,0,0]))
if not np.any(u_hat1):
    u_hat1 = np.cross(n_hat1,np.array([0,1,0]))

v_hat1 = np.cross(n_hat1,u_hat1)

n_hat2 = pt1_2/deputy_rad

u_hat2 = np.cross(n_hat2,np.array([1,0,0]))
if not np.any(u_hat2):
    u_hat2 = np.cross(n_hat2,np.array([0,1,0]))

v_hat2 = np.cross(n_hat2,u_hat2)

def P(t,u,v):
    P = np.cos(t)*u + np.sin(t)*v
    return P

proj1 = o_0 - np.dot(o_0,n_hat1)/np.linalg.norm(n_hat1)**2*n_hat1
proj2 = o_0 - np.dot(o_0,n_hat2)/np.linalg.norm(n_hat2)**2*n_hat2


def looper():
    for t1 in np.linspace(4.2,5.0,400):
        print(t1)
        for t2 in np.linspace(t1-0.5,t1+0.5,500):
            if t1 != t2:
                o_1 = P(t1,u_hat1,v_hat1)
                o_2 = P(t2,u_hat1,v_hat2)
                o_1 /= np.linalg.norm(o_1)
                o_2 /= np.linalg.norm(o_2)
                inc_1 = np.arccos(o_1[2])
                phi_1= np.arctan2(o_1[1], o_1[0])
                inc_2 = np.arccos(o_2[2])
                phi_2= np.arctan2(o_2[1], o_2[0])

                #Next, compute the orbital rotation matrices for these two other satellites.
                Rs = [R_0]
                #poffsets = [0, -delta, delta]
                radii_fraction = deputy_rad/R_orb
                radii = [1,radii_fraction,radii_fraction]
                for phi, inc in zip([phi_1, phi_2], [inc_1, inc_2]):
                    R_phi = np.array([[np.cos(phi), -np.sin(phi),0],[np.sin(phi), np.cos(phi), 0],[0,0,1]])
                    R_th = np.array([[np.cos(inc),0,np.sin(inc)], [0, 1, 0], [-np.sin(inc), 0, np.cos(inc)]])
                    Rs.append(R_phi.dot(R_th).dot(np.linalg.inv(R_phi)))

                init_pos_diff_1 = np.dot(np.linalg.inv(Rs[1]),pt1_1) - xyzc[0]*radii_fraction
                init_pos_diff_2 = np.dot(np.linalg.inv(Rs[2]),pt1_2) - xyzc[0]*radii_fraction

                if init_pos_diff_1[1]>=0:
                    phase1 = np.linalg.norm(init_pos_diff_1)/deputy_rad
                else:
                    phase1 = -np.linalg.norm(init_pos_diff_1)/deputy_rad
                if init_pos_diff_2[1]>=0:
                    phase2 = np.linalg.norm(init_pos_diff_2)/deputy_rad
                else:
                    phase2 = -np.linalg.norm(init_pos_diff_2)/deputy_rad
                poffsets = [0, phase1, phase2]

                #Use the Rotation matrices to find the orbital positions.
                global xyzo
                for ix, R in enumerate(Rs):
                    for iy in range(n_p):
                        xyzo[ix,iy] = np.dot(R, ((xyzc[iy]*radii[ix]/R_orb)/radii + poffsets[ix])*radii)

                #Lets compute dot products for the full orbit
                eps = 2
                eps2 = 0.5
                eps3 = 0.5
                for ix in range(0,n_p,50):
                    sep1 = xyzo[1,ix] - xyzo[0,ix]
                    sep2 = xyzo[2,ix] - xyzo[0,ix]
                    if abs(np.dot(s_hat,sep1)) > eps or abs(np.dot(s_hat,sep2)) > eps \
                    or abs(phi_1-phi0) > eps2 or abs(inc_1-inc0) > eps3 \
                    or abs(phi_2-phi0) > eps2 or abs(inc_2-inc0) > eps3:
                        break
                    elif ix == n_p-1:
                        print("BREAK")
                        print(abs(np.dot(s_hat,sep1)),abs(np.dot(s_hat,sep2)))
                        return (phase1,phase2,phi0,phi_1,phi_2,inc0,inc_1,inc_2)
    return (0,0,0,0,0,0)




            #print(sep1,sep2,s_hat)
            #print("Dot products with star vector: {:6.1f} {:6.1f}".format(np.dot(s_hat, sep1), np.dot(s_hat, sep2)))

(phase1,phase2,phi0,phi_1,phi_2,inc0,inc_1,inc_2) = looper()
#Next, compute the orbital rotation matrices for these two other satellites.
Rs = [R_0]
poffsets = [0,phase1,phase2]
print(phi0, inc0)
for phi, inc in zip([phi_1, phi_2], [inc_1, inc_2]):
    print(phi, inc)
    R_phi = np.array([[np.cos(phi), -np.sin(phi),0],[np.sin(phi), np.cos(phi), 0],[0,0,1]])
    R_th = np.array([[np.cos(inc),0,np.sin(inc)], [0, 1, 0], [-np.sin(inc), 0, np.cos(inc)]])
    Rs.append(R_phi.dot(R_th).dot(np.linalg.inv(R_phi)))

#Use the Rotation matrices to find the orbital positions.
for ix, R in enumerate(Rs):
    for iy in range(n_p):
        xyzo[ix,iy] = np.dot(R, xyzc[iy])
"""
for ix in range(n_p):
    sep1 = xyzo[1,ix] - xyzo[0,ix]
    sep2 = xyzo[2,ix] - xyzo[0,ix]
    sep1 /= np.linalg.norm(sep1)
    sep2 /= np.linalg.norm(sep2)
    print(sep1,sep2,s_hat)
    print("Dot products with star vector: {:6.1f} {:6.1f}".format(np.dot(s_hat, sep1), np.dot(s_hat, sep2)))
"""

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
        sat_xyz = [np.interp( (sat_phase) % (2*np.pi), phase, xyz[:,ii]) for ii in range(3)]

        #If in foreground or more than R_earth away in (x,z) plane, plot.
        if (sat_xyz[1] > 0) | (np.sqrt(sat_xyz[0]**2 + sat_xyz[2]**2) > R_e):
            plt.plot(sat_xyz[0] + R_e, sat_xyz[2] + R_e,point)
    plt.tight_layout()
    #plt.savefig("pngs/orb{:03d}.png".format(im_ix))
    plt.pause(.1)
