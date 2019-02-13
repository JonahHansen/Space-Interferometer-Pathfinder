
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
#import astropy.constants as const
import quaternions as qt
from mpl_toolkits import mplot3d

from astropy import units as u
from poliastro.bodies import Earth,Moon, Sun
#from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
#from poliastro.plotting import plot, OrbitPlotter3D, OrbitPlotter
import perturbations as ptb

plt.ion()

alt = 500   #In km
R_e = Earth.R.to(u.km).value  #In km
n_p = 100 #Number of phases

#Orbital inclination
inc_0 = np.radians(29) #49
#Longitude of the Ascending Node
Om_0 = np.radians(-32) #-30

#Stellar vector
ra = np.radians(4) #23
dec = np.radians(45)#43

#The distance to the other satellites in km
b = 350

period = 2703.0 #In minutes.
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

#Perturbations
perturbs = [0]

xyzp = np.zeros( (3,n_p,3) )

tof = (period * u.min).to(u.s).value

for i in range(3):
    orb = ptb.from_pos_to_orbit(xyzo[i,0],xyzo[i,1],n_p,period)
    rr, vv = cowell(orb, np.linspace(0, tof, n_p), ad=ptb.perturbations, index_ls = perturbs)
    xyzp[i] = rr
    
#xyzo = xyzo*u.m.to(u.km)
ax = plt.axes(projection='3d')
ax.plot3D(xyzo[0,:,0],xyzo[0,:,1],xyzo[0,:,2],'b-')
ax.plot3D(xyzp[0,:,0],xyzp[0,:,1],xyzp[0,:,2],'r-')