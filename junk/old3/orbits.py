"""ORBIT Class - Calculates orbits and converts between orbital frames"""

import numpy as np
import astropy.constants as const
import modules.quaternions as qt

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
class ECI_orbit:

    def __init__(self, R_orb, delta_r_max, inc_0, Om_0, ra, dec):

        self.Om_0 = Om_0
        self.inc_0 = inc_0

        self.delta_r_max = delta_r_max

        self.R_orb = R_orb #Orbital radius
        self.period = 2*np.pi*np.sqrt((R_orb)**3/const.GM_earth).value #Period in seconds.
        self.ang_vel = 2*np.pi/self.period #Angular velocity of orbit

        #Star vector
        self.s_hat = np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])

        #Initial axis unit vectors
        xaxis = np.array([1,0,0])
        yaxis = np.array([0,1,0])
        zaxis = np.array([0,0,1])

        #U-V plane vectors
        self.u_hat = np.cross(self.s_hat,zaxis)
        self.v_hat = np.cross(self.s_hat,self.u_hat)

        #Precession
        J2 = 0.00108263
        self.w_p = -3/2*const.R_earth.value**2/R_orb**2*J2*self.ang_vel*np.cos(inc_0)

        #Quaternion rotation of chief orbit
        q_Om = qt.to_q(zaxis,Om_0)
        q_inc = qt.to_q(yaxis,-inc_0)
        self.q0 = qt.comb_rot(qt.comb_rot(qt.conjugate(q_Om),q_inc),q_Om)

        #Angular momentum vector of chief satellite
        self.h_0 = qt.rotate(zaxis,self.q0)

        #New coord system:
        z_hat = self.h_0 #In direction of angular momentum


        x = self.s_hat-z_hat*(np.dot(self.s_hat,z_hat)) #Projection of the star vector on the orbital plane

        if (x == np.array([0.,0.,0.])).all():
            if (z_hat == np.array([1.,0.,0.])).all():
                x = np.array([0.,1.,0.])
            else:
                x = np.cross(z_hat,np.array([1.,0.,0.]))

        x_hat = x/np.linalg.norm(x)
        y_hat = np.cross(z_hat,x_hat) #Remaining orthogonal vector

        #Angle between angular momentum vector and star (checks are for precision errors):
        dot = np.dot(z_hat,self.s_hat)
        if dot < -1.:
            dot = -1.
        elif dot > 1.:
            dot = 1.

        theta = np.arccos(dot)

        psi = self.delta_r_max*np.cos(theta)/self.R_orb #Angle between chief and deputy WRT Earth
        omega = -np.arctan(self.delta_r_max/self.R_orb*np.sin(theta)) #Amount of rotation

        #Define deputy orbital planes in terms of a rotation of the chief satellite
        axis1 = -np.cos(psi)*x_hat - np.sin(psi)*y_hat #Axis of rotation
        q_phase1 = qt.to_q(z_hat,psi) #Rotate in phase
        q_plane1 = qt.to_q(axis1,omega) #Rotate around axis
        self.q1 = qt.comb_rot(q_phase1,q_plane1) #Combine

        #Same as above but for the second deputy
        axis2 = -np.cos(psi)*x_hat + np.sin(psi)*y_hat
        q_phase2 = qt.to_q(z_hat,-psi)
        q_plane2 = qt.to_q(axis2,-omega)
        self.q2 = qt.comb_rot(q_phase2,q_plane2)

    """ Find u and v vectors given the deputy state vectors"""
    def uv(self,ECI_dep1,ECI_dep2):
        sep = ECI_dep2.pos - ECI_dep1.pos #Baseline vector
        u = np.dot(sep,self.u_hat)
        v = np.dot(sep,self.v_hat)
        return np.array([u,v])

class Satellite:
    def __init__(self,pos,vel,q):
        self.pos = pos #position
        self.vel = vel #velocity
        self.q = q #orbit rotation quaternion
        self.state = np.concatenate((self.pos,self.vel)) #state vector

    """ Orbital elements from state vector """
    def orbit_elems(self):
        h = np.cross(self.pos,self.vel) #Angular momentum vector
        n = np.cross(np.array([0,0,1]),h)
        i = np.arccos(h[2]/np.linalg.norm(h)) #Inclination
        omega = np.arccos(n[0]/np.linalg.norm(n)) #Longitude of the ascending node
        if n[1] < 0:
            omega = 360 - omega
        return i,omega


class Chief(Satellite):
    def __init__(self,ECI,t,precession=False):
        Satellite.__init__(self,np.zeros(3),np.zeros(3),ECI.q0)

        self.ang_vel = ECI.ang_vel
        self.R_orb = ECI.R_orb

        phase = t*self.ang_vel

        z_hat = np.array([0,0,1])

        #Take into account precession
        if precession:
            del_Om = ECI.w_p*t #Amount of precession
            q_del_Om = qt.to_q(zaxis,del_Om)
            self.q = qt.comb_rot(self.q,q_del_Om) #New chief quaternion

        #Angular momentum vector of chief satellite
        self.h_0 = qt.rotate(z_hat,self.q)

        #Base orbit from phase
        self.pos[0] = np.cos(phase) * self.R_orb
        self.pos[1] = np.sin(phase) * self.R_orb
        self.vel[0] = -np.sin(phase) * self.R_orb * self.ang_vel
        self.vel[1] = np.cos(phase) * self.R_orb * self.ang_vel

        #Rotate in orbit
        self.pos = qt.rotate(self.pos,self.q)
        self.vel = qt.rotate(self.vel,self.q)
        self.state = np.concatenate((self.pos,self.vel))

        rho_hat = self.pos/np.linalg.norm(self.pos) #Position unit vector (rho)
        xi_hat = self.vel/np.linalg.norm(self.vel) #Velocity unit vector (xi)
        eta_hat = np.cross(rho_hat,xi_hat) #Angular momentum vector (eta)
        self.mat = np.array([rho_hat,xi_hat,eta_hat]) #LVLH rotation matrix


class Deputy(Satellite):
    def __init__(self,pos,vel,q):
        Satellite.__init__(self,pos,vel,q)


    def to_LVLH(self,chief):
        non_zero_pos = np.dot(chief.mat,self.pos) #Position in LVLH, origin at centre of Earth
        pos = non_zero_pos - np.dot(chief.mat,chief.pos) #Position, origin at chief spacecraft
        omega = np.array([0,0,chief.ang_vel]) #Angular momentum vector in LVLH frame
        vel = np.dot(chief.mat,self.vel) - np.cross(omega,non_zero_pos) #Velocity, including rotating frame
        return Deputy(pos,vel,self.q)

    """ Takes a given state vector in LVLH coordinates and converts to ECI """
    """ Requires chief state and the change of basis matrix """
    def to_ECI(self,chief):
        inv_rotmat = np.linalg.inv(chief.mat) #LVLH to ECI change of basis matrix
        pos = np.dot(inv_rotmat,self.pos) + chief.pos #ECI position
        omega = np.array([0,0,chief.ang_vel]) #Angular momentum vector
        #Velocity in ECI frame, removing the rotation of the LVLH frame
        vel = np.dot(inv_rotmat,(self.vel + np.cross(omega,np.dot(chief.mat,pos))))
        return Deputy(pos,vel,self.q)

def init_deputy(ECI,chief,n,precession):

    if precession:
        #New coord system:
        z_hat = ECI.h_0 #In direction of angular momentum

        x = ECI.s_hat-z_hat*(np.dot(ECI.s_hat,z_hat)) #Projection of the star vector on the orbital plane

        if (x == np.array([0.,0.,0.])).all():
            if (z_hat == np.array([1.,0.,0.])).all():
                x = np.array([0.,1.,0.])
            else:
                x = np.cross(z_hat,np.array([1.,0.,0.]))

        x_hat = x/np.linalg.norm(x)
        y_hat = np.cross(z_hat,x_hat) #Remaining orthogonal vector

        #Angle between angular momentum vector and star (checks are for precision errors):
        dot = np.dot(z_hat,ECI.s_hat)
        if dot < -1.:
            dot = -1.
        elif dot > 1.:
            dot = 1.

        theta = np.arccos(dot)

        psi = self.delta_r_max*np.cos(theta)/self.R_orb #Angle between chief and deputy WRT Earth
        omega = -np.arctan(self.delta_r_max/self.R_orb*np.sin(theta)) #Amount of rotation

        if n == 1:
            #Define deputy orbital planes in terms of a rotation of the chief satellite
            axis = -np.cos(psi)*x_hat - np.sin(psi)*y_hat #Axis of rotation
            q_phase = qt.to_q(z_hat,psi) #Rotate in phase
            q_plane = qt.to_q(axis,omega) #Rotate around axis
            q = qt.comb_rot(q_phase,q_plane) #Combine

        elif n == 2:
            #Same as above but for the second deputy
            axis = -np.cos(psi)*x_hat + np.sin(psi)*y_hat
            q_phase = qt.to_q(z_hat,-psi)
            q_plane = qt.to_q(axis,-omega)
            q = qt.comb_rot(q_phase,q_plane)

    else:
        #Init satellite with the correct quaternion
        if n == 1:
            q = ECI.q1
        elif n == 2:
            q = ECI.q2
        else:
            raise Exception("Bad Deputy number")

    return Deputy(qt.rotate(chief.pos,q),qt.rotate(chief.vel,q),q)
