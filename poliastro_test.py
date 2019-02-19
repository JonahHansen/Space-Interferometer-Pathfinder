
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody.propagation import cowell
from poliastro.twobody import Orbit
import quaternions as qt
import perturbations as ptb
from frame import orbits_to_LVLH, orbits_to_baseline

plt.ion()

alt = 1000e3 #In km

n_p = 1000 #Number of phases

#Orbital inclination
inc_0 = np.radians(0) #49
#Longitude of the Ascending Node
Om_0 = np.radians(34) #-30

#Stellar vector
ra = np.radians(52) #23
dec = np.radians(45)#43

#The distance to the other satellites in km
b = 0.3*1e3

#Perturbations (see module)
perturbs = [3,4]
j_date = 2454283.0 * u.day

drag_coeff = [0,0,0]
front_area = [0,0,0]
mass = [0,0,0]
atm_scale_height = [0,0,0]
exponent_density_prefactor = [0,0,0]

rad_pressure_coeff = [0,0,0]
effective_spacecraft_area = [0,0,0]

#------------------------------------------------------------------------------------------
#Orbital radius the sum of earth radius and altitude
R_e = Earth.R.to(u.m).value #In m
R_orb = R_e + alt
#Orbital phase, i.e. mean longitude. n_p the number of phases
phase = np.linspace(0, 2*np.pi, n_p)

period = 2*np.pi*np.sqrt((R_orb)**3/const.GM_earth).value #In seconds.

ang_vel = 2*np.pi/period

lines = ['r:', 'g:', 'g:']
points = ['r.', 'g.', 'g.']

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



#Array of orbits, both normal and perturbed
xyzf = np.zeros( (6,n_p,3) )

t_f = period

epoch = Time(j_date, format='jd', scale = 'tdb')

times = np.linspace(0, t_f, n_p) #Times for orbit

if 3 in perturbs:
    print("BUILDING MOON EPHEM")
    moon = ptb.moon_ephem(t_f,j_date)
else:
    moon = 0

if 4 in perturbs or 5 in perturbs:
    print("BUILDING SUN EPHEM")
    sun = ptb.sun_ephem(t_f,j_date)
else:
    sun = 0

#C_D (float) – dimensionless drag coefficient ()
        #A (float) – frontal area of the spacecraft (km^2)
        #m (float) – mass of the spacecraft (kg)
        #H0 (float) – atmospheric scale height (km)
        #rho0

xyzf[0:3] = xyzo

for i in range(3):
    #Orbit from poliastro
    #orb = ptb.from_pos_to_orbit(xyzo[i,0],xyzo[i,1],n_p,period)
    orb = Orbit.from_vectors(Earth, xyzo[i,0]*u.m, uvwo[i,0]*u.m / u.s, epoch = epoch)

    #Integrate orbit with given peturbations and append to array
    rr, vv = cowell(orb, times, ad=ptb.perturbations, index_ls = perturbs, C_D = drag_coeff[i],
                    A = front_area[i], m = mass[i], H0 = atm_scale_height[i], rho0 = exponent_density_prefactor[i],
                    moon = moon, sun = sun, C_R = rad_pressure_coeff[i], A2 = effective_spacecraft_area[i])
    xyzf[i+3] = rr
    xyzf[i+3] *= 1e3 #To m

#Arrays of LVLH positions, both normal and perturbed
#0-2 = Normal, 3-5 = Perturbed
lvlhf = np.zeros( (6,n_p,3) )
for i in range(n_p):
    lvlhf[0:3,i] = orbits_to_LVLH(xyzf[0,i],[xyzf[1,i],xyzf[2,i]],q_0)
    lvlhf[3:6,i] = orbits_to_LVLH(xyzf[3,i],[xyzf[4,i],xyzf[5,i]],q_0)

#Arrays of Baseline positions, both normal and perturbed
#0-2 = Normal, 3-5 = Perturbed
basef = np.zeros( (6,n_p,3) )
for i in range(n_p):
    basef[:,i] = orbits_to_baseline(xyzf[0,i],xyzf[1,i],xyzf[2,i],s_hat,xyzf[3:6,i])

#Effects on separation of deputy spacecraft due to peturbation
#Separation of both

del_deputy_xyz0 = xyzf[2]-xyzf[1]-(xyzf[5]-xyzf[4])
del_deputy_xyz1 = xyzf[1]-xyzf[0]-(xyzf[4]-xyzf[3])
del_deputy_xyz2 = xyzf[2]-xyzf[0]-(xyzf[5]-xyzf[3])

del_deputy_lvlh0 = lvlhf[2]-lvlhf[1]-(lvlhf[5]-lvlhf[4])
del_deputy_lvlh1 = lvlhf[1]-lvlhf[0]-(lvlhf[4]-lvlhf[3])
del_deputy_lvlh2 = lvlhf[2]-lvlhf[0]-(lvlhf[5]-lvlhf[3])

del_deputy_base0 = basef[2]-basef[1]-(basef[5]-basef[4])
del_deputy_base1 = basef[1]-basef[0]-(basef[4]-basef[3])
del_deputy_base2 = basef[2]-basef[0]-(basef[5]-basef[3])

del_deputy_mag0 = np.array([np.linalg.norm(i) for i in del_deputy_xyz0])
del_deputy_mag1 = np.array([np.linalg.norm(i) for i in del_deputy_xyz1])
del_deputy_mag2 = np.array([np.linalg.norm(i) for i in del_deputy_xyz2])

"""
delt = 6000 #Times to plot in s
del_deputy_xyz = del_deputy_xyz[0:int(delt*n_p/period)]*1e3
del_deputy_lvlh = del_deputy_lvlh[0:int(delt*n_p/period)]*1e3
del_deputy_mag = del_deputy_mag[0:int(delt*n_p/period)]*1e3
times = times[0:int(delt*n_p/period)]
"""
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
#ax1.plot3D(del_deputy_xyz0[:,0],del_deputy_xyz0[:,1],del_deputy_xyz0[:,2],'b-')
ax1.plot3D(del_deputy_xyz1[:,0],del_deputy_xyz1[:,1],del_deputy_xyz1[:,2],'r-')
ax1.plot3D(del_deputy_xyz2[:,0],del_deputy_xyz2[:,1],del_deputy_xyz2[:,2],'g-')
ax1.set_xlabel('Delta x (m)')
ax1.set_ylabel('Delta y (m)')
ax1.set_zlabel('Delta z (m)')
ax1.set_title('Effect of perturbation on deputy separation in geocentric cartesian coordinates')
set_axes_equal(ax1)

### 2D Geocentric Cartesian plots: Effect of perturbation on deputy separation against time for each axis ###
fig2, [ax2_1, ax2_2, ax2_3] = plt.subplots(3, 1, sharex=True, sharey=True)
#ax2_1.plot(times,del_deputy_xyz0[:,0],"b-")
ax2_1.plot(times,del_deputy_xyz1[:,0],"r-")
ax2_1.plot(times,del_deputy_xyz2[:,0],"g-")
#ax2_2.plot(times,del_deputy_xyz0[:,1],"b-")
ax2_2.plot(times,del_deputy_xyz1[:,1],"r-")
ax2_2.plot(times,del_deputy_xyz2[:,1],"g-")
#ax2_3.plot(times,del_deputy_xyz0[:,2],"b-")
ax2_3.plot(times,del_deputy_xyz1[:,2],"r-")
ax2_3.plot(times,del_deputy_xyz2[:,2],"g-")
ax2_1.set_ylabel('Delta x (m)')
ax2_2.set_ylabel('Delta y (m)')
ax2_3.set_ylabel('Delta z (m)')
ax2_3.set_xlabel('Time (s)')
ax2_1.set_title('Effect of perturbation on deputy separation in geocentric cartesian coordinates')

### 3D LVLH plot of the effect of perturbations on deputy separation###
plt.figure(3)
ax3 = plt.axes(projection='3d')
ax3.set_aspect('equal')
#ax3.plot3D(del_deputy_lvlh0[:,0],del_deputy_lvlh0[:,1],del_deputy_lvlh0[:,2],'b-')
ax3.plot3D(del_deputy_lvlh1[:,0],del_deputy_lvlh1[:,1],del_deputy_lvlh1[:,2],'r-')
ax3.plot3D(del_deputy_lvlh2[:,0],del_deputy_lvlh2[:,1],del_deputy_lvlh2[:,2],'g-')
ax3.set_xlabel('Delta r (m)')
ax3.set_ylabel('Delta v (m)')
ax3.set_zlabel('Delta h (m)')
ax3.set_title('Effect of perturbation on deputy separation in LVLH coordinates')
set_axes_equal(ax3)

### 2D LVLH plots: Effect of perturbation on deputy separation against time for each axis ###
fig4, [ax4_1, ax4_2, ax4_3] = plt.subplots(3, 1, sharex=True, sharey=True)
#ax4_1.plot(times,del_deputy_lvlh0[:,0],"b-")
ax4_1.plot(times,del_deputy_lvlh1[:,0],"r-")
ax4_1.plot(times,del_deputy_lvlh2[:,0],"g-")
#ax4_2.plot(times,del_deputy_lvlh0[:,1],"b-")
ax4_2.plot(times,del_deputy_lvlh1[:,1],"r-")
ax4_2.plot(times,del_deputy_lvlh2[:,1],"g-")
#ax4_3.plot(times,del_deputy_lvlh0[:,2],"b-")
ax4_3.plot(times,del_deputy_lvlh1[:,2],"r-")
ax4_3.plot(times,del_deputy_lvlh2[:,2],"g-")
ax4_1.set_ylabel('Delta r (m)')
ax4_2.set_ylabel('Delta v (m)')
ax4_3.set_ylabel('Delta h (m)')
ax4_3.set_xlabel('Time (s)')
ax4_1.set_title('Effect of perturbation on deputy separation in LVLH coordinates')

### 3D Baseline plot of the effect of perturbations on deputy separation###
plt.figure(5)
ax5 = plt.axes(projection='3d')
ax5.set_aspect('equal')
#ax5.plot3D(del_deputy_base0[:,0],del_deputy_base0[:,1],del_deputy_base0[:,2],'b-')
ax5.plot3D(del_deputy_base1[:,0],del_deputy_base1[:,1],del_deputy_base1[:,2],'r-')
ax5.plot3D(del_deputy_base2[:,0],del_deputy_base2[:,1],del_deputy_base2[:,2],'g-')
ax5.set_xlabel('Delta a (m)')
ax5.set_ylabel('Delta b (m)')
ax5.set_zlabel('Delta s (m)')
ax5.set_title('Effect of perturbation on deputy separation in Baseline coordinates')
set_axes_equal(ax5)

### 2D Baseline plots: Effect of perturbation on deputy separation against time for each axis ###
fig6, [ax6_1, ax6_2, ax6_3] = plt.subplots(3, 1, sharex=True, sharey=True)
#ax6_1.plot(times,del_deputy_base0[:,0],"b-")
ax6_1.plot(times,del_deputy_base1[:,0],"r-")
ax6_1.plot(times,del_deputy_base2[:,0],"g-")
#ax6_2.plot(times,del_deputy_base0[:,1],"b-")
ax6_2.plot(times,del_deputy_base1[:,1],"r-")
ax6_2.plot(times,del_deputy_base2[:,1],"g-")
#ax6_3.plot(times,del_deputy_base0[:,2],"b-")
ax6_3.plot(times,del_deputy_base1[:,2],"r-")
ax6_3.plot(times,del_deputy_base2[:,2],"g-")
ax6_1.set_ylabel('Delta a (m)')
ax6_2.set_ylabel('Delta b (m)')
ax6_3.set_ylabel('Delta s (m)')
ax6_3.set_xlabel('Time (s)')
ax6_1.set_title('Effect of perturbation on deputy separation in Baseline coordinates')

plt.figure(7)
#plt.plot(times,del_deputy_mag0,'b-')
plt.plot(times,del_deputy_mag1,'r-')
plt.plot(times,del_deputy_mag2,'g-')
plt.ylabel("Magnitude of separation (m)")
plt.xlabel("Time (s)")
plt.title("Magnitude of change in separation due to peturbation against time")

#plt.show()