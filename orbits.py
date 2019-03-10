"""ORBIT Classes - Calculates orbits and converts between orbital frames"""

import numpy as np
import astropy.constants as const
import quaternions as qt

"""
 Parent class of orbit - requires number of phases and radius of orbit
 to initialise
"""
class sat_orbit:
    def __init__(self, n_p, R_orb):

        self.n_p = n_p #Number of phases
        self.phase = np.linspace(0, 2*np.pi, n_p) #Phase array

        self.R_orb = R_orb #Orbital radius
        self.period = 2*np.pi*np.sqrt((R_orb)**3/const.GM_earth).value #Period in seconds.
        self.ang_vel = 2*np.pi/self.period #Angular velocity of orbit

        #Initialise position and velocity vectors
        self.chief_pos = np.zeros((n_p,3))
        self.chief_vel = np.zeros((n_p,3))
        self.deputy1_pos = np.zeros((n_p,3))
        self.deputy1_vel = np.zeros((n_p,3))
        self.deputy2_pos = np.zeros((n_p,3))
        self.deputy2_vel = np.zeros((n_p,3))
        
        #Initialise deputy separation vectors (from chief satellite)
        self.deputy1_pos_sep = np.zeros((n_p,3))
        self.deputy1_vel_sep = np.zeros((n_p,3))
        self.deputy2_pos_sep = np.zeros((n_p,3))
        self.deputy2_vel_sep = np.zeros((n_p,3))

    #Helper functions to give the state vector of a given satellite
    def chief_state_vec(self):
        return np.array([self.chief_pos[:,0],self.chief_pos[:,1],self.chief_pos[:,2],
                         self.chief_vel[:,0],self.chief_vel[:,1],self.chief_vel[:,2]])

    def deputy1_state_vec(self):
        return np.array([self.deputy1_pos[:,0],self.deputy1_pos[:,1],self.deputy1_pos[:,2],
                         self.deputy1_vel[:,0],self.deputy1_vel[:,1],self.deputy1_vel[:,2]])

    def deputy2_state_vec(self):
        return np.array([self.deputy2_pos[:,0],self.deputy2_pos[:,1],self.deputy2_pos[:,2],
                         self.deputy2_vel[:,0],self.deputy2_vel[:,1],self.deputy2_vel[:,2]])

    def deputy1_sep_state_vec(self):
        return np.array([self.deputy1_pos_sep[:,0],self.deputy1_pos_sep[:,1],self.deputy1_pos_sep[:,2],
                         self.deputy1_vel_sep[:,0],self.deputy1_vel_sep[:,1],self.deputy1_vel_sep[:,2]])

    def deputy2_sep_state_vec(self):
        return np.array([self.deputy2_pos_sep[:,0],self.deputy2_pos_sep[:,1],self.deputy2_pos_sep[:,2],
                         self.deputy2_vel_sep[:,0],self.deputy2_vel_sep[:,1],self.deputy2_vel_sep[:,2]])

"""
ECI (Earth Centred Inertial) Orbit class: Use this to first calculate the orbit from scratch.
Origin at the centre of the Earth, cartesian, satellite starts on positive x axis

Parameters:
n_p = number of phases
R_orb = radius of orbit
delta_r_max = maximum separation of deputies from chief
inc_0,Om_0 = orientation of chief orbit
ra,dec = position of star to be observed
"""
class ECI_orbit(sat_orbit):

    def __init__(self, n_p, R_orb, delta_r_max, inc_0, Om_0, ra, dec):
        sat_orbit.__init__(self, n_p, R_orb)

        self.Om_0 = Om_0
        self.inc_0 = inc_0

        self.delta_r_max = delta_r_max
        
        #Star vector
        self.s_hat = [np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)]

        #Calculate position and velcity for each phase
        for i in range(self.n_p):
            self.chief_pos[i,0] = np.cos(self.phase[i]) * self.R_orb
            self.chief_pos[i,1] = np.sin(self.phase[i]) * self.R_orb
            self.chief_vel[i,0] = -np.sin(self.phase[i]) * self.R_orb * self.ang_vel
            self.chief_vel[i,1] = np.cos(self.phase[i]) * self.R_orb * self.ang_vel

        #Initial axis unit vectors
        xaxis = np.array([1,0,0])
        yaxis = np.array([0,1,0])
        zaxis = np.array([0,0,1])

        #Quaternion rotation of chief orbit
        q_Om = qt.to_q(zaxis,Om_0)
        q_inc = qt.to_q(yaxis,inc_0)
        self.q0 = qt.comb_rot(qt.comb_rot(qt.conjugate(q_Om),q_inc),q_Om)

        #Cartesian points on the rotated orbit
        self.chief_pos = qt.rotate_points(self.chief_pos,self.q0)
        self.chief_vel = qt.rotate_points(self.chief_vel,self.q0)

        #Angular momentum vector of chief satellite
        h_0 = qt.rotate(zaxis,self.q0)

        #New coord system:
        z_hat = h_0/np.linalg.norm(h_0) #In direction of angular momentum
        print(z_hat)
        y = self.s_hat-z_hat*(np.dot(self.s_hat,z_hat)) #Projection of the star vector on the orbital plane
        print(y)
        y_hat = y/np.linalg.norm(y)
        print(y_hat)
        x_hat = np.cross(z_hat,y_hat) #Remaining orthogonal vector

        #Angle between angular momentum vector and star (checks are for precision errors):
        dot = np.dot(z_hat,self.s_hat)
        if dot < -1.:
            dot = -1.
        elif dot > 1.:
            dot = 1.

        theta = np.arccos(dot)

        psi = self.delta_r_max*np.cos(theta)/self.R_orb #Angle between chief and deputy WRT Earth

        #Define deputy orbital planes in terms of a rotation of the chief satellite
        axis1 = -np.cos(psi)*y_hat + np.sin(psi)*x_hat #Axis of rotation
        omega1 = np.arctan(self.delta_r_max/self.R_orb*np.sin(theta)) #Amount of rotation
        q_phase1 = qt.to_q(z_hat,-psi) #Rotate in phase
        q_plane1 = qt.to_q(axis1,omega1) #Rotate around axis
        self.q1 = qt.comb_rot(q_phase1,q_plane1) #Combine

        #Same as above but for the second deputy
        axis2 = -np.cos(-psi)*y_hat + np.sin(-psi)*x_hat
        omega2 = np.arctan(-self.delta_r_max/self.R_orb*np.sin(theta))
        q_phase2 = qt.to_q(z_hat,psi)
        q_plane2 = qt.to_q(axis2,omega2)
        self.q2 = qt.comb_rot(q_phase2,q_plane2)

        #Rotate the chiefs orbit
        self.deputy1_pos = qt.rotate_points(self.chief_pos,self.q1)
        self.deputy1_vel = qt.rotate_points(self.chief_vel,self.q1)
        self.deputy2_pos = qt.rotate_points(self.chief_pos,self.q2)
        self.deputy2_vel = qt.rotate_points(self.chief_vel,self.q2)

        self.deputy1_pos_sep = (self.deputy1_pos - self.chief_pos)
        self.deputy2_pos_sep = (self.deputy2_pos - self.chief_pos)
        self.deputy1_vel_sep = (self.deputy1_vel - self.chief_vel)
        self.deputy2_vel_sep = (self.deputy2_vel - self.chief_vel)
        
    def chief_position(self,t):
        phase = t*self.angvel

"""
LVLH Orbit class:
Origin at the chief spacecraft
r = position direction
v = velocity direction
h = OAM direction

Parameters:
n_p = number of phases
R_orb = radius of orbit
ECI = ECI orbit to convert to LVLH
"""
class LVLH_orbit(sat_orbit):

    def __init__(self, n_p, R_orb, ECI):

        sat_orbit.__init__(self, n_p, R_orb)
        self.s_hats = np.zeros((n_p,3)) #List of star vectors in LVLH frame

        for ix in range(ECI.n_p):

            h_hat = qt.rotate(np.array([0,0,1]),ECI.q0) #Angular momentum vector (rotated "z" axis)
            r_hat = ECI.chief_pos[ix]/np.linalg.norm(ECI.chief_pos[ix]) #Position vector pointing away from the centre of the Earth
            v_hat = np.cross(h_hat,r_hat) #Velocity vector pointing counter-clockwise

            rot_mat = np.array([r_hat,v_hat,h_hat]) #Rotation matrix from three unit vectors

            #New vectors, position set relative to the chief's position
            self.chief_pos[ix] = np.dot(rot_mat,ECI.chief_pos[ix])-np.dot(rot_mat,ECI.chief_pos[ix])
            self.chief_vel[ix] = np.dot(rot_mat,ECI.chief_vel[ix])
            self.deputy1_pos[ix] = np.dot(rot_mat,ECI.deputy1_pos[ix])-np.dot(rot_mat,ECI.chief_pos[ix])
            self.deputy1_vel[ix] = np.dot(rot_mat,ECI.deputy1_vel[ix])
            self.deputy2_pos[ix] = np.dot(rot_mat,ECI.deputy2_pos[ix])-np.dot(rot_mat,ECI.chief_pos[ix])
            self.deputy2_vel[ix] = np.dot(rot_mat,ECI.deputy2_vel[ix])
            self.s_hats[ix] = np.dot(rot_mat,ECI.s_hat)
            
        self.deputy1_pos_sep = (self.deputy1_pos - self.chief_pos)
        self.deputy2_pos_sep = (self.deputy2_pos - self.chief_pos)
        self.deputy1_vel_sep = (self.deputy1_vel - self.chief_vel)
        self.deputy2_vel_sep = (self.deputy2_vel - self.chief_vel)

"""
Baseline Orbit class:
Origin at the chief spacecraft
b = baseline direction towards deputy 2
s = star direction
k = other direction

Parameters:
n_p = number of phases
R_orb = radius of orbit
ECI = ECI orbit to convert to Baseline
"""
class Baseline_orbit(sat_orbit):

    def __init__(self, n_p, R_orb, ECI):

        sat_orbit.__init__(self, n_p, R_orb)

        for ix in range(ECI.n_p):
            
            b_hat = (ECI.deputy2_pos[ix] - ECI.chief_pos[ix])/np.linalg.norm(ECI.deputy2_pos[ix] - ECI.chief_pos[ix]) #Direction along baseline
            k_hat = np.cross(ECI.s_hat,b_hat) #Other direction

            rot_mat = np.array([k_hat,b_hat,ECI.s_hat]) #Create rotation matrix

            #New vectors, position set relative to the chief's position
            self.chief_pos[ix] = np.dot(rot_mat,ECI.chief_pos[ix])-np.dot(rot_mat,ECI.chief_pos[ix])
            self.chief_vel[ix] = np.dot(rot_mat,ECI.chief_vel[ix])
            self.deputy1_pos[ix] = np.dot(rot_mat,ECI.deputy1_pos[ix])-np.dot(rot_mat,ECI.chief_pos[ix])
            self.deputy1_vel[ix] = np.dot(rot_mat,ECI.deputy1_vel[ix])
            self.deputy2_pos[ix] = np.dot(rot_mat,ECI.deputy2_pos[ix])-np.dot(rot_mat,ECI.chief_pos[ix])
            self.deputy2_vel[ix] = np.dot(rot_mat,ECI.deputy2_vel[ix])

        self.deputy1_pos_sep = (self.deputy1_pos - self.chief_pos)
        self.deputy2_pos_sep = (self.deputy2_pos - self.chief_pos)
        self.deputy1_vel_sep = (self.deputy1_vel - self.chief_vel)
        self.deputy2_vel_sep = (self.deputy2_vel - self.chief_vel)