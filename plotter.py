"""
A script to plot orbits simply for illustration

Note that cartopy and Basemap both should work here.

7 Feb 2019: Version 2.0 by Jonah Hansen. Works for circular orbits.
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import astropy.constants as const
import astropy.units as u
from poliastro.bodies import Earth
from poliastro.twobody.propagation import cowell
import perturbations as ptb
import quaternions as qt
from frame import orbits_to_LVLH

plt.ion()

alt = 500.0e3   #In km
R_e = Earth.R.to(u.km).value*1000  #In km
n_p = 1000 #Number of phases

#Orbital inclination
inc_0 = np.radians(29) #49
#Longitude of the Ascending Node
Om_0 = np.radians(-32) #-30

#Stellar vector
ra = np.radians(4) #23
dec = np.radians(45)#43

#The distance to the other satellites in km
b = 350e3

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

#Initial axis unit vectors
xaxis = np.array([1,0,0])
yaxis = np.array([0,1,0])
zaxis = np.array([0,0,1])

#Quaternion rotation, using Mike's "No phase difference" rotation!
q_Om = qt.to_q(zaxis,Om_0)
q_inc = qt.to_q(yaxis,inc_0)
q_0 = qt.comb_rot(qt.comb_rot(qt.conjugate(q_Om),q_inc),q_Om)

#Cartesian points on the rotated orbit
xyzo[0] = qt.rotate_points(xyzc,q_0)

#Angular momentum vector of chief satellite
h_0 = qt.rotate(zaxis,q_0)

#Vector pointing to the star, from right ascension and declination
s_hat = [np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)]

#New coord system:
z_hat = h_0/np.linalg.norm(h_0) #In direction of angular momentum
y = s_hat-z_hat*(np.dot(s_hat,z_hat)) #Projection of the star vector on the orbital plane
y_hat = y/np.linalg.norm(y)
x_hat = np.cross(z_hat,y_hat) #Remaining orthogonal vector

#Angle between angular momentum vector and star:
theta =np.arccos(np.dot(z_hat,s_hat))

psi = b/R_orb #Angle between chief and deputy WRT Earth

#Define deputy orbital planes in terms of a rotation of the chief satellite
axis1 = -np.cos(psi)*y_hat + np.sin(psi)*x_hat #Axis of rotation
angle1 = np.arctan(psi*np.tan(theta)) #Amount of rotation
q_phase1 = qt.to_q(z_hat,-psi) #Rotate in phase
q_plane1 = qt.to_q(axis1,angle1) #Rotate around axis
q_orb1 = qt.comb_rot(q_phase1,q_plane1) #Combine

#Same as above but for the second deputy
axis2 = -np.cos(-psi)*y_hat + np.sin(-psi)*x_hat
angle2 = np.arctan(-psi*np.tan(theta))
q_phase2 = qt.to_q(z_hat,psi)
q_plane2 = qt.to_q(axis2,angle2)
q_orb2 = qt.comb_rot(q_phase2,q_plane2)

#Rotate the chiefs orbit
xyzo[1] = qt.rotate_points(xyzo[0],q_orb1)
xyzo[2] = qt.rotate_points(xyzo[0],q_orb2)

"""
def c_lvlh(i):
    c,d1,d2,s = lvlh.orbits_to_LVLH(xyzo[0,i],xyzo[1,i],xyzo[2,i],s_hat,q_0)
    return c,d1,d2,s

#Lets compute dot products for the full orbit
for ix in range(0,n_p,10):
    sep1 = xyzo[1,ix] - xyzo[0,ix]
    sep2 = xyzo[2,ix] - xyzo[0,ix]
    print(np.dot(sep1,s_hat),np.dot(sep2,s_hat))
    print("Angles to correct plane: {:6.3f} {:6.3f}".format(np.arcsin(np.dot(s_hat, sep1/np.linalg.norm(sep1))), np.arcsin(np.dot(s_hat, sep2/np.linalg.norm(sep2)))))
"""
"""
#Perturbations (see module)
perturbs = [1]

#Array of perturbed orbits
xyzp = np.zeros( (3,n_p,3) )

t_f = period
times = np.linspace(0, t_f, n_p) #Times for orbit

for i in range(3):
    #Orbit from poliastro
    orb = ptb.from_pos_to_orbit(xyzo[i,0],xyzo[i,1],n_p,period)

    #Integrate orbit with given peturbations and append to array
    rr, vv = cowell(orb, times, ad=ptb.perturbations, index_ls = perturbs)
    xyzp[i] = rr
"""
#Make pretty plots.
pos_ls = [] #list of positions
for im_ix, sat_phase in enumerate(np.linspace(np.pi,3.*np.pi,15)): #np.pi, 31*np.pi,450))
#for sat_phase in np.linspace(np.pi*1.45,np.pi*1.5,2):
    plt.clf()
    plt.subplot(1, 2, 1)
    map = Basemap(projection='ortho',lat_0=0,lon_0=180 - np.degrees(sat_phase*period/24/60),resolution='l')
    map.bluemarble(scale=0.4)
    plt.axis([-0.1*R_orb, 2.1*R_orb, -0.0*R_orb, 2.0*R_orb])
    xyz_ls = []
    #Find non-vignetted parts (two vectors)
    for xyz, point, line in zip(xyzo, points, lines):
        visible = (xyz[:,1] > 0) | (np.sqrt(xyz[:,0]**2 + xyz[:,2]**2) > R_e)
        visible = np.concatenate(([False],visible, [False]))
        out_of_eclipse = np.where(visible[1:] & np.logical_not(visible[:-1]))[0]
        in_to_eclipse = np.where(np.logical_not(visible[1:]) & visible[:-1])[0]
        for oute, ine in zip(out_of_eclipse, in_to_eclipse):
            plt.plot(xyz[oute:ine+1,0] + R_e, xyz[oute:ine+1,2] + R_e,line)

        #Interpolate to current time.
        sat_xyz = [np.interp( (sat_phase) % (2*np.pi), phase, xyz[:,ii]) for ii in range(3)]
        xyz_ls.append(sat_xyz)

        #If in foreground or more than R_earth away in (x,z) plane, plot.
        if (sat_xyz[1] > 0) | (np.sqrt(sat_xyz[0]**2 + sat_xyz[2]**2) > R_e):
            plt.plot(sat_xyz[0] + R_e, sat_xyz[2] + R_e,point)

    plt.tight_layout()
    plt.subplot(3, 3, 6)

    km = 1e-3

    plt.xlim(-2*b*km,2*b*km)
    plt.ylim(-2*b*km,2*b*km)

    #LVLH positions
    lvlh_vecs = orbits_to_LVLH(xyz_ls[0],[xyz_ls[1],xyz_ls[2],s_hat],q_0)

    #Star vector
    s = lvlh_vecs[3]
    plt.arrow(0,0,b*km*s[1],b*km*s[2],width=b*km/40,color='k')

    #List of positions
    pos_ls.append([lvlh_vecs[0],lvlh_vecs[1],lvlh_vecs[2]])
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
