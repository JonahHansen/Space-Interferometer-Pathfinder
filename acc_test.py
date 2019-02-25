
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
from multiprocessing import Pool

plt.ion()

alt = 1000e3 #In km

len_ix = 360
len_iy = 360

ix_vals = np.linspace(0,90,len_ix)
iy_vals = np.linspace(-90,90,len_iy)

#The max distance to the other satellites in km
delta_r_max = 0.3*1e3

#Perturbations (see module)
perturbs = [1]
j_date = 2454283.0 * u.day #Epoch

#Parameters for peturbations

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

period = 2*np.pi*np.sqrt((R_orb)**3/const.GM_earth).value #In seconds.

n_p = int(period) #Each phase iteration = 1 second
#Orbital phase, i.e. mean longitude. n_p the number of phases
phase = np.linspace(0, 2*np.pi, n_p)

ang_vel = 2*np.pi/period

def worker(ix):
    #Orbital inclination
    inc_0 = np.radians(ix_vals[ix]) #49
    #Longitude of the Ascending Node
    Om_0 = np.radians(0) #-30

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

    z_hat = h_0/np.linalg.norm(h_0) #In direction of angular momentum

    t_f = period
    epoch = Time(j_date, format='jd', scale = 'tdb') #Epoch
    times = np.linspace(0, t_f, n_p) #Times for orbit

    #Build ephemeris if required
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

    result = np.zeros((len_iy,8))

    for iy in range(len_iy):

        print(ix,iy)
        #Stellar vector
        ra = np.radians(0) #23
        dec = np.radians(iy_vals[iy])#43

        #Vector pointing to the star, from right ascension and declination
        s_hat = [np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)]

        #New coord system:
        y = s_hat-z_hat*(np.dot(s_hat,z_hat)) #Projection of the star vector on the orbital plane
        y_hat = y/np.linalg.norm(y)
        x_hat = np.cross(z_hat,y_hat) #Remaining orthogonal vector

        #Angle between angular momentum vector and star:
        theta = np.arccos(np.dot(z_hat,s_hat))

        psi = delta_r_max*np.cos(theta)/R_orb #Angle between chief and deputy WRT Earth

        #Define deputy orbital planes in terms of a rotation of the chief satellite
        axis1 = -np.cos(psi)*y_hat + np.sin(psi)*x_hat #Axis of rotation
        omega1 = np.arctan(delta_r_max/R_orb*np.sin(theta)) #Amount of rotation
        q_phase1 = qt.to_q(z_hat,-psi) #Rotate in phase
        q_plane1 = qt.to_q(axis1,omega1) #Rotate around axis
        q_orb1 = qt.comb_rot(q_phase1,q_plane1) #Combine

        #Same as above but for the second deputy
        axis2 = -np.cos(-psi)*y_hat + np.sin(-psi)*x_hat
        omega2 = np.arctan(-delta_r_max/R_orb*np.sin(theta))
        q_phase2 = qt.to_q(z_hat,psi)
        q_plane2 = qt.to_q(axis2,omega2)
        q_orb2 = qt.comb_rot(q_phase2,q_plane2)

        #Rotate the chiefs orbit
        xyzo[1] = qt.rotate_points(xyzo[0],q_orb1)
        xyzo[2] = qt.rotate_points(xyzo[0],q_orb2)
        uvwo[1] = qt.rotate_points(uvwo[0],q_orb1)
        uvwo[2] = qt.rotate_points(uvwo[0],q_orb2)

        #Array of orbits, both normal and perturbed
        xyzf = np.zeros( (6,n_p,3) )

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


        #Arrays of Baseline positions, both normal and perturbed
        #0-2 = Normal, 3-5 = Perturbed
        basef = np.zeros( (6,n_p,3) )
        for i in range(n_p):
            basef[:,i] = orbits_to_baseline(xyzf[0,i],xyzf[1,i],xyzf[2,i],s_hat,xyzf[3:6,i])

        #Effects on separation of deputy spacecraft due to peturbation
        #Separation of both

        s1 = (basef[4]-basef[3])[:,2]
        s2 = (basef[5]-basef[3])[:,2]
        delta_b = np.abs((basef[4]-basef[3])[:,1])-np.abs((basef[5]-basef[3])[:,1])

        ## --------------------------------------------------------------------##

        def acc(pos):
            vel = np.gradient(pos, edge_order=2)
            acc = np.gradient(vel, edge_order=2)
            return acc

        acc_s1 = acc(s1)
        acc_s2 = acc(s2)
        acc_delta_b = acc(delta_b)

        max_acc_s1 = max(acc_s1)
        max_acc_s2 = max(acc_s2)
        max_acc_delta_b = max(acc_delta_b)

        delta_v_s1 = np.trapz(np.abs(acc_s1))
        delta_v_s2 = np.trapz(np.abs(acc_s2))
        delta_v_delta_b = np.trapz(np.abs(acc_delta_b))

        result[iy]  = np.array([ix,iy,max_acc_s1,max_acc_s2,max_acc_delta_b,delta_v_s1,delta_v_s2,delta_v_delta_b])

    return result

p = Pool(processes=25)
result = p.map(worker,range(len_ix))
result = np.array(result)
np.save("2acc_variance_alt1000_ra0.npy",result)

alt = 500e3 #In km
R_orb = R_e + alt

period = 2*np.pi*np.sqrt((R_orb)**3/const.GM_earth).value #In seconds.

n_p = int(period) #Each phase iteration = 1 second
#Orbital phase, i.e. mean longitude. n_p the number of phases
phase = np.linspace(0, 2*np.pi, n_p)

ang_vel = 2*np.pi/period

p = Pool(processes=25)
result = p.map(worker,range(len_ix))
result = np.array(result)
np.save("2acc_variance_alt500_ra0.npy",result)
