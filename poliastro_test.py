
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from astropy import units as u
from poliastro.bodies import Earth, Moon, Sun
from poliastro.twobody.propagation import cowell
from poliastro.twobody import Orbit
import quaternions as qt
import perturbations as ptb
from LVLH import orbits_to_LVLH

#plt.ion()

alt = 500 #In km
R_e = Earth.R.to(u.km).value  #In km
n_p = 1000 #Number of phases

#Orbital inclination
inc_0 = np.radians(60) #49
#Longitude of the Ascending Node
Om_0 = np.radians(34) #-30

#Stellar vector
ra = np.radians(52) #23
dec = np.radians(45)#43

#The distance to the other satellites in km
b = 3

#period = 95.5*60
lines = ['r:', 'g:', 'g:']
points = ['r.', 'g.', 'g.']

#------------------------------------------------------------------------------------------
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt
#Orbital phase, i.e. mean longitude. n_p the number of phases
phase = np.linspace(0, 2*np.pi, n_p)

period = 2*np.pi*np.sqrt((R_orb*1e3)**3/const.GM_earth).value #In seconds.

ang_vel = 2*np.pi/period

#Central spacecraft Cartesian coordinates (and velocities) for a circular orbit in the x,y plane.
xyzc = np.zeros( (n_p,3) )
uvwc = np.zeros( (n_p,3) )
for i in range(n_p):
    xyzc[i,0] = np.cos(phase[i]) * R_orb
    xyzc[i,1] = np.sin(-phase[i]) * R_orb
    uvwc[i,0] = -np.sin(phase[i]) * R_orb * ang_vel
    uvwc[i,1] = -np.cos(phase[i]) * R_orb * ang_vel

#Cartesian coordinates for all 3 spacecraft
xyzo = np.zeros( (3,n_p,3) )
uvwo = np.zeros( (3,n_p,3) )

#Initial axis unit vectors
xaxis = np.array([1,0,0])
yaxis = np.array([0,1,0])
zaxis = np.array([0,0,1])

#Quaternion rotation, using Mike's "No phase difference" rotation!
q_Om = qt.to_q(zaxis,Om_0)
q_inc = qt.to_q(xaxis,inc_0) #xaxis to be consistent with poliastro
q_0 = qt.comb_rot(qt.comb_rot(qt.conjugate(q_Om),q_inc),q_Om)

#Cartesian points on the rotated orbit
xyzo[0] = qt.rotate_points(xyzc,q_0)
uvwo[0] = qt.rotate_points(uvwc,q_0)

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
theta = np.arccos(np.dot(z_hat,s_hat))

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
uvwo[1] = qt.rotate_points(uvwo[0],q_orb1)
uvwo[2] = qt.rotate_points(uvwo[0],q_orb2)


#Perturbations (see module)
perturbs = [1]

#Array of perturbed orbits
xyzp = np.zeros( (3,n_p,3) )

t_f = period
times = np.linspace(0, t_f, n_p) #Times for orbit

for i in range(3):
    #Orbit from poliastro
    #orb = ptb.from_pos_to_orbit(xyzo[i,0],xyzo[i,1],n_p,period)
    orb = Orbit.from_vectors(Earth, xyzo[i,0]*u.km, uvwo[i,0]*u.km / u.s)

    #Integrate orbit with given peturbations and append to array
    rr, vv = cowell(orb, times, ad=ptb.perturbations, index_ls = perturbs)
    xyzp[i] = rr

#Arrays of LVLH positions
lvlho = np.zeros( (3,n_p,3) )
lvlhp = np.zeros( (3,n_p,3) )
for i in range(n_p):
    lvlho[:,i] = orbits_to_LVLH(xyzo[0,i],[xyzo[1,i],xyzo[2,i]],q_0)
    lvlhp[:,i] = orbits_to_LVLH(xyzp[0,i],[xyzp[1,i],xyzp[2,i]],q_0)

#Effects on separation of deputy spacecraft due to peturbation

del_deputy_xyz = xyzo[1]-xyzo[2]-(xyzp[1]-xyzp[2])
del_deputy_lvlh = lvlho[1]-lvlho[2]-(lvlhp[1]-lvlhp[2])

del_deputy_mag = np.array([np.linalg.norm(i) for i in del_deputy_xyz])

delt = 6000 #Times to plot in s
del_deputy_xyz = del_deputy_xyz[0:int(delt*n_p/period)]*1e3
del_deputy_lvlh = del_deputy_lvlh[0:int(delt*n_p/period)]*1e3
del_deputy_mag = del_deputy_mag[0:int(delt*n_p/period)]*1e3
times = times[0:int(delt*n_p/period)]

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

"""

### 3D Geocentric Cartesian plot of the effect of perturbations ###
plt.figure(1)
ax1 = plt.axes(projection='3d')
ax1.set_aspect('equal')
ax1.plot3D(xyzo[0,:,0]-xyzp[0,:,0],xyzo[0,:,1]-xyzp[0,:,1],xyzo[0,:,2]-xyzp[0,:,2],'b-')
ax1.plot3D(xyzo[1,:,0]-xyzp[1,:,0],xyzo[1,:,1]-xyzp[1,:,1],xyzo[1,:,2]-xyzp[1,:,2],'g-')
ax1.plot3D(xyzo[2,:,0]-xyzp[2,:,0],xyzo[2,:,1]-xyzp[2,:,1],xyzo[2,:,2]-xyzp[2,:,2],'r-')
ax1.set_xlabel('Delta x (km)')
ax1.set_ylabel('Delta y (km)')
ax1.set_zlabel('Delta z (km)')
ax1.set_title('Effect of perturbation on orbit in geocentric cartesian coordinates')
set_axes_equal(ax1)

### 2D Geocentric Cartesian plots: Effect of perturbation against time for each axis ###
fig2, [ax2_1, ax2_2, ax2_3] = plt.subplots(3, 1, sharex=True, sharey=True)
ax2_1.plot(times,xyzo[0,:,0]-xyzp[0,:,0],"b-")
ax2_1.plot(times,xyzo[1,:,0]-xyzp[1,:,0],"r-")
ax2_1.plot(times,xyzo[2,:,0]-xyzp[2,:,0],"g-")

ax2_2.plot(times,xyzo[0,:,1]-xyzp[0,:,1],"b-")
ax2_2.plot(times,xyzo[1,:,1]-xyzp[1,:,1],"r-")
ax2_2.plot(times,xyzo[2,:,1]-xyzp[2,:,1],"g-")

ax2_3.plot(times,xyzo[0,:,2]-xyzp[0,:,2],"b-")
ax2_3.plot(times,xyzo[1,:,2]-xyzp[1,:,2],"r-")
ax2_3.plot(times,xyzo[2,:,2]-xyzp[2,:,2],"g-")

ax2_1.set_ylabel('Delta x (km)')
ax2_2.set_ylabel('Delta y (km)')
ax2_3.set_ylabel('Delta z (km)')
ax2_3.set_xlabel('Time (s)')
ax2_1.set_title('Effect of perturbation on orbit in geocentric cartesian coordinates')


### 3D LVLH plot of the effect of perturbations ###
plt.figure(3)
ax3 = plt.axes(projection='3d')
ax3.set_aspect('equal')
ax3.plot3D(lvlho[0,:,0]-lvlhp[0,:,0],lvlho[0,:,1]-lvlhp[0,:,1],lvlho[0,:,2]-lvlhp[0,:,2],'b-')
ax3.plot3D(lvlho[1,:,0]-lvlhp[1,:,0],lvlho[1,:,1]-lvlhp[1,:,1],lvlho[1,:,2]-lvlhp[1,:,2],'g-')
ax3.plot3D(lvlho[2,:,0]-lvlhp[2,:,0],lvlho[2,:,1]-lvlhp[2,:,1],lvlho[2,:,2]-lvlhp[2,:,2],'r-')
ax3.set_xlabel('Delta r (km)')
ax3.set_ylabel('Delta v (km)')
ax3.set_zlabel('Delta h (km)')
ax3.set_title('Effect of perturbation on orbit in LVLH coordinates')
set_axes_equal(ax3)

### 2D LVLH plots: Effect of perturbation against time for each axis ###
fig4, [ax4_1, ax4_2, ax4_3] = plt.subplots(3, 1, sharex=True, sharey=True)
ax4_1.plot(times,lvlho[0,:,0]-lvlhp[0,:,0],"b-")
ax4_1.plot(times,lvlho[1,:,0]-lvlhp[1,:,0],"r-")
ax4_1.plot(times,lvlho[2,:,0]-lvlhp[2,:,0],"g-")

ax4_2.plot(times,lvlho[0,:,1]-lvlhp[0,:,1],"b-")
ax4_2.plot(times,lvlho[1,:,1]-lvlhp[1,:,1],"r-")
ax4_2.plot(times,lvlho[2,:,1]-lvlhp[2,:,1],"g-")

ax4_3.plot(times,lvlho[0,:,2]-lvlhp[0,:,2],"b-")
ax4_3.plot(times,lvlho[1,:,2]-lvlhp[1,:,2],"r-")
ax4_3.plot(times,lvlho[2,:,2]-lvlhp[2,:,2],"g-")
ax4_1.set_ylabel('Delta r (km)')
ax4_2.set_ylabel('Delta v (km)')
ax4_3.set_ylabel('Delta h (km)')
ax4_3.set_xlabel('Time (s)')
ax4_1.set_title('Effect of perturbation on orbit in LVLH coordinates')

"""

### 3D Geocentric Cartesian plot of the effect of perturbations on deputy separation###
plt.figure(1)
ax1 = plt.axes(projection='3d')
ax1.set_aspect('equal')
ax1.plot3D(del_deputy_xyz[:,0],del_deputy_xyz[:,1],del_deputy_xyz[:,2],'k-')
ax1.set_xlabel('Delta x (m)')
ax1.set_ylabel('Delta y (m)')
ax1.set_zlabel('Delta z (m)')
ax1.set_title('Effect of perturbation on deputy separation in geocentric cartesian coordinates')
set_axes_equal(ax1)

### 2D Geocentric Cartesian plots: Effect of perturbation on deputy separation against time for each axis ###
fig2, [ax2_1, ax2_2, ax2_3] = plt.subplots(3, 1, sharex=True, sharey=True)
ax2_1.plot(times,del_deputy_xyz[:,0],"b-")
ax2_2.plot(times,del_deputy_xyz[:,1],"r-")
ax2_3.plot(times,del_deputy_xyz[:,2],"g-")
ax2_1.set_ylabel('Delta x (m)')
ax2_2.set_ylabel('Delta y (m)')
ax2_3.set_ylabel('Delta z (m)')
ax2_3.set_xlabel('Time (s)')
ax2_1.set_title('Effect of perturbation on deputy separation in geocentric cartesian coordinates')

### 3D LVLH plot of the effect of perturbations on deputy separation###
plt.figure(3)
ax3 = plt.axes(projection='3d')
ax3.set_aspect('equal')
ax3.plot3D(del_deputy_lvlh[:,0],del_deputy_lvlh[:,1],del_deputy_lvlh[:,2],'k-')
ax3.set_xlabel('Delta r (m)')
ax3.set_ylabel('Delta v (m)')
ax3.set_zlabel('Delta h (m)')
ax3.set_title('Effect of perturbation on deputy separation in LVLH coordinates')
set_axes_equal(ax3)

### 2D LVLH plots: Effect of perturbation on deputy separation against time for each axis ###
fig4, [ax4_1, ax4_2, ax4_3] = plt.subplots(3, 1, sharex=True, sharey=True)
ax4_1.plot(times,del_deputy_lvlh[:,0],"b-")
ax4_2.plot(times,del_deputy_lvlh[:,1],"r-")
ax4_3.plot(times,del_deputy_lvlh[:,2],"g-")
ax4_1.set_ylabel('Delta r (m)')
ax4_2.set_ylabel('Delta v (m)')
ax4_3.set_ylabel('Delta h (m)')
ax4_3.set_xlabel('Time (s)')
ax4_1.set_title('Effect of perturbation on deputy separation in LVLH coordinates')

plt.figure(5)
plt.plot(times,del_deputy_mag,'k-')
plt.ylabel("Magnitude of separation (m)")
plt.xlabel("Time (s)")
plt.title("Magnitude of change in separation due to peturbation against time")

plt.show()
