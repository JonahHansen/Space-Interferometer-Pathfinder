"""ORBIT Classes - Calculates orbits and converts between orbital frames"""

import numpy as np
import astropy.constants as const
import quaternions as qt

"""
 Parent class of orbit - requires number of phases and radius of orbit
 to initialise

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

    def __init__(self, R_orb, delta_r_max, inc_0, Om_0, ra, dec):
        
        self.Om_0 = Om_0
        self.inc_0 = inc_0

        self.delta_r_max = delta_r_max
        
        self.R_orb = R_orb #Orbital radius
        self.period = 2*np.pi*np.sqrt((R_orb)**3/const.GM_earth).value #Period in seconds.
        self.ang_vel = 2*np.pi/self.period #Angular velocity of orbit
    
        #Star vector
        self.s_hat = [np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)]

        #Initial axis unit vectors
        xaxis = np.array([1,0,0])
        yaxis = np.array([0,1,0])
        zaxis = np.array([0,0,1])

        #Quaternion rotation of chief orbit
        q_Om = qt.to_q(zaxis,Om_0)
        q_inc = qt.to_q(yaxis,inc_0)
        self.q0 = qt.comb_rot(qt.comb_rot(qt.conjugate(q_Om),q_inc),q_Om)

        #Angular momentum vector of chief satellite
        self.h_0 = qt.rotate(zaxis,self.q0)

        #New coord system:
        z_hat = self.h_0 #In direction of angular momentum
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
        
    def chief_state(self,t):
        phase = t*self.ang_vel
        
        pos = np.zeros((0,3))
        vel = np.zeros((0,3))
        
        pos[0] = np.cos(phase) * self.R_orb
        pos[1] = np.sin(phase) * self.R_orb
        vel[0] = -np.sin(phase) * self.R_orb * self.ang_vel
        vel[1] = np.cos(phase) * self.R_orb * self.ang_vel
        
        pos = qt.rotate(pos,self.q0)
        vel = qt.rotate(vel,self.q0)
        return pos, vel

    def deputy1_state(self,chief_pos,chief_vel)
        pos = np.rotate(chief_pos,self.q1)
        vel = np.rotate(chief_vel,self.q1)
        return pos, vel
        
    def deputy2_state(self,chief_pos,chief_vel)
        pos = np.rotate(chief_pos,self.q2)
        vel = np.rotate(chief_vel,self.q2)
        return pos, vel
            
    def to_LVLH_mat(self,chief_pos):
        r_hat = chief_pos/np.linalg.norm(chief_pos)
        v_hat = np.cross(self.h_0,r_hat)
        rot_mat = np.array([r_hat,v_hat,self.h_0])
        return rot_mat
        
    def to_Baseline_mat(self,chief_pos,deputy_pos):
        b_hat = (deputy_pos-chief_pos)/np.linalg.norm(deputy_pos-chief_pos)
        k_hat = np.cross(self.s_hat,b_hat)
        rot_mat = np.array([k_hat,b_hat,self.s_hat])
        return rot_mat
