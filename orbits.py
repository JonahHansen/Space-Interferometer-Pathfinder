"""ORBIT Class - Calculates orbits and converts between orbital frames"""

import numpy as np
import astropy.constants as const
import quaternions as qt

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
        self.s_hat = [np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)]

        #Initial axis unit vectors
        xaxis = np.array([1,0,0])
        yaxis = np.array([0,1,0])
        zaxis = np.array([0,0,1])

        #U-V plane vectors
        self.u_hat = np.cross(self.s_hat,zaxis)
        self.v_hat = np.cross(self.s_hat,self.u_hat)

        #Quaternion rotation of chief orbit
        q_Om = qt.to_q(zaxis,Om_0)
        q_inc = qt.to_q(yaxis,-inc_0)
        self.q0 = qt.comb_rot(qt.comb_rot(qt.conjugate(q_Om),q_inc),q_Om)

        #Angular momentum vector of chief satellite
        self.h_0 = qt.rotate(zaxis,self.q0)

        #New coord system:
        z_hat = self.h_0 #In direction of angular momentum


        y = self.s_hat-z_hat*(np.dot(self.s_hat,z_hat)) #Projection of the star vector on the orbital plane

        if (y == np.array([0.,0.,0.])).all():
            if (z_hat == np.array([1.,0.,0.])).all():
                y = np.array([0.,1.,0.])
            else:
                y = np.cross(z_hat,np.array([1.,0.,0.]))

        y_hat = y/np.linalg.norm(y)
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

    """Calculate chief state at a given time t"""
    def chief_state(self,t):
        phase = t*self.ang_vel

        pos = np.zeros(3)
        vel = np.zeros(3)

        #Base orbit from phase
        pos[0] = np.cos(phase) * self.R_orb
        pos[1] = np.sin(phase) * self.R_orb
        vel[0] = -np.sin(phase) * self.R_orb * self.ang_vel
        vel[1] = np.cos(phase) * self.R_orb * self.ang_vel

        #Rotate in orbit
        pos = qt.rotate(pos,self.q0)
        vel = qt.rotate(vel,self.q0)
        return np.append(pos,vel)

    """Calculate deputy 1 state for a given chief state"""
    def deputy1_state(self,chief_state):
        pos = qt.rotate(chief_state[0:3],self.q1)
        vel = qt.rotate(chief_state[3:],self.q1)
        return np.append(pos,vel)

    """Calculate deputy 2 state for a given chief state"""
    def deputy2_state(self,chief_state):
        pos = qt.rotate(chief_state[0:3],self.q2)
        vel = qt.rotate(chief_state[3:],self.q2)
        return np.append(pos,vel)

    """ Create change of basis matrix - requires chief position """
    def to_LVLH_mat(self,chief_state):
        chief_pos = chief_state[:3]
        r_hat = chief_pos/np.linalg.norm(chief_pos)
        v_hat = np.cross(self.h_0,r_hat)
        rot_mat = np.array([r_hat,v_hat,self.h_0])
        return rot_mat

    """ Takes a given state vector in ECI coordinates and converts to LVLH """
    def ECI_to_LVLH_state(self,ECI_chief,rot_mat,ECI_state):
        non_zero_pos = np.dot(rot_mat,ECI_state[0:3]) #Position in LVLH, origin at centre of Earth
        pos = non_zero_pos - np.dot(rot_mat,ECI_chief[0:3]) #Position, origin at chief spacecraft
        omega = np.array([0,0,self.ang_vel]) #Angular momentum vector in LVLH frame
        vel = np.dot(rot_mat,ECI_state[3:]) - np.cross(omega,non_zero_pos) #Velocity, including rotating frame
        return np.append(pos,vel)

    """ Takes a given state vector in LVLH coordinates and converts to ECI """
    def LVLH_to_ECI_state(self,ECI_chief,rot_mat,LVLH_state):
        inv_rotmat = np.linalg.inv(rot_mat) #LVLH to ECI change of basis matrix
        pos = np.dot(inv_rotmat,LVLH_state[:3]) + ECI_chief[:3] #ECI position
        omega = np.array([0,0,self.ang_vel]) #Angular momentum vector
        #Velocity in ECI frame, removing the rotation of the LVLH frame
        vel = np.dot(inv_rotmat,(LVLH_state[3:] + np.cross(omega,np.dot(rot_mat,pos))))
        return np.append(pos,vel)

    """ Finds the vector pointing away from the sun """
    def find_anti_sun_vector(self,t):
        oblq = np.radians(23.4) #Obliquity of the ecliptic
        Earth_sun_ang_vel = 2*np.pi/(365.25*24*60*60) #Angular velocity of the Earth around the Sun
        phase = t*Earth_sun_ang_vel
        pos = np.array([np.cos(phase)*np.cos(oblq),np.sin(phase),np.cos(phase)*np.sin(oblq)])
        return pos

    """ Find u and v vectors """
    def uv(self,ECI_dep1,ECI_dep2):
        sep = ECI_dep2[:3] - ECI_dep1[:3] #Baseline vector
        u = np.dot(sep,self.u_hat)*self.u_hat
        v = np.dot(sep,self.v_hat)*self.v_hat
        return np.array([u,v])

    """ Orbital elements from state vector """
    def orbit_elems(self,state):
        h = np.cross(state[:3],state[3:])
        n = np.cross(np.array([0,0,1]),h)
        i = np.arccos(h[2]/np.linalg.norm(h))
        omega = np.arccos(n[0]/np.linalg.norm(n))
        if n[1] < 0:
            omega = 360 - omega
        return i,omega

    def asdasd(self,ECI2):
        combq1 = qt.comb_rot(qt.conjugate(self.q1),ECI2.q1)
        v_q1,theta_q1 = qt.from_q(combq1)
        combq2 = qt.comb_rot(qt.conjugate(self.q2),ECI2.q2)
        v_q2,theta_q2 = qt.from_q(combq2)


        times = np.linspace(0,ECI2.period,10000)
        pos = np.zeros(3)
        eps = 0.0001
        for t in times:
            phase = t*self.ang_vel
            pos[0] = np.cos(phase)
            pos[1] = np.sin(phase)
            pos_11 = qt.rotate(pos,self.q1)
            pos_12 = qt.rotate(pos,ECI2.q1)
            pos_21 = qt.rotate(pos,self.q2)
            pos_22 = qt.rotate(pos,ECI2.q2)

            if np.linalg.norm(pos_11 - v_q1) < eps:
                t_11 = t
            if np.linalg.norm(pos_12 - v_q1) < eps:
                t_12 = t
            if np.linalg.norm(pos_21 - v_q2) < eps:
                t_21 = t
            if np.linalg.norm(pos_22 - v_q2) < eps:
                t_22 = t

            print(np.linalg.norm(pos_11 - v_q1))
        mu = const.GM_earth.value

        def vis_viva(r,a):
            return np.sqrt(mu*(2/r - 1/a))

        del_t1 = t_12 - t_11
        T1 = del_t1 + self.period
        a1 = (mu*(T1/(2*np.pi))**2)**(1/3)
        del_v1 = vis_viva(self.R_orb,a1) - vis_viva(self.R_orb,self.R_orb)

        del_t2 = t_22 - t_21
        T2 = del_t2 + self.period
        a2 = (mu*(T2/(2*np.pi))**2)**(1/3)
        del_v2 = vis_viva(self.R_orb,a2) - vis_viva(self.R_orb,self.R_orb)

        vel_11 = deputy1_state(chief_state(t_11))[:3]
        vel_12 = deputy1_state(chief_state(t_12))[:3]
        vel_21 = deputy1_state(chief_state(t_21))[:3]
        vel_22 = deputy1_state(chief_state(t_22))[:3]

        del_v1 += np.linalg.norm(vel_12-vel_11)
        del_v2 += np.linalg.norm(vel_22-vel_21)
        return delv_1, delv_2
