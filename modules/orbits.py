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
class Reference_orbit:

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
        if (self.s_hat == zaxis).all():
           self.u_hat = np.array([1,0,0])
        else:
           self.u_hat = np.cross(self.s_hat,zaxis)
        self.v_hat = np.cross(self.s_hat,self.u_hat)
        self.UVmat = np.array([self.u_hat,self.v_hat,self.s_hat]) #LVLH rotation matrix

        #J2 Perturbation constants
        J2 = 0.00108263
        Sch_s = 3/8*const.R_earth.value**2/R_orb**2*J2*(1+3*np.cos(2*inc_0))
        self.Sch_c = np.sqrt(1+Sch_s)
        self.Sch_k = self.ang_vel*self.Sch_c + 3*self.ang_vel*J2*const.R_earth.value**2/(2*R_orb**2)*(np.cos(inc_0)**2)

        #Quaternion rotation of reference (chief) orbit
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
        
    def ref_orbit_pos(self,t,precession=False):
        
        if precession:
            J2 = 0.00108263
            sq_mu = np.sqrt(const.GM_earth.value)
            R_e = const.R_earth.value
            
            factor = 3*sq_mu*J2*R_e**2/(2*self.R_orb**(3.5))*np.cos(self.inc_0)
            
            i = self.inc_0 - factor/self.Sch_k*np.sin(self.inc_0)
            Om = self.Om_0 - factor*t
            th = self.Sch_k*t
            dot_i = 0
            dot_Om = -factor
            dot_th = self.Sch_k
            
            pos = np.zeros(3)
            vel = np.zeros(3)
            
            pos[0] = self.R_orb*(np.cos(Om)*np.cos(th)-np.sin(Om)*np.sin(th)*np.cos(i))
            pos[1] = self.R_orb*(np.sin(Om)*np.cos(th)+np.cos(Om)*np.sin(th)*np.cos(i))
            pos[2] = self.R_orb*(np.sin(th)*np.sin(i))
            
            vel[0] = self.R_orb*(-np.sin(Om)*np.cos(th)*dot_Om - np.cos(Om)*np.sin(th)*dot_th -
                                  np.cos(Om)*np.sin(th)*np.cos(i)*dot_Om - np.sin(Om)*np.cos(th)*np.cos(i)*dot_th)
            vel[1] = self.R_orb*(np.cos(Om)*np.cos(th)*dot_Om - np.sin(Om)*np.sin(th)*dot_th -
                                 np.sin(Om)*np.sin(th)*np.cos(i)*dot_Om + np.cos(Om)*np.cos(th)*np.cos(i)*dot_th)
            vel[2] = self.R_orb*(np.cos(th)*np.sin(i)*dot_th)
            
        else:
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
            
        rho_hat = pos/np.linalg.norm(pos) #Position unit vector (rho)
        xi_hat = vel/np.linalg.norm(vel) #Velocity unit vector (xi)
        eta_hat = np.cross(rho_hat,xi_hat) #Angular momentum vector (eta)
        LVLH_mat = np.array([rho_hat,xi_hat,eta_hat]) #LVLH rotation matrix
    
        b_hat = np.cross(rho_hat,self.s_hat) #Baseline unit vector
        o_hat = np.cross(self.s_hat,b_hat) #Other unit vector
        Base_mat = np.array([b_hat,o_hat,self.s_hat]) #Baseline rotation matrix
        
        return pos,vel,LVLH_mat,Base_mat



class Satellite:
    def __init__(self,pos,vel,time,reference):
        self.pos = pos #position
        self.vel = vel #velocity
        self.time = time
        self.state = np.concatenate((self.pos,self.vel)) #state vector
        self.reference = reference
        
    """ Orbital elements from state vector """
    def orbit_elems(self):
        h = np.cross(self.pos,self.vel) #Angular momentum vector
        n = np.cross(np.array([0,0,1]),h)
        i = np.arccos(h[2]/np.linalg.norm(h)) #Inclination
        omega = np.arccos(n[0]/np.linalg.norm(n)) #Longitude of the ascending node
        if n[1] < 0:
            omega = 360 - omega
        return i,omega



class ECI_Sat(Satellite):
    def __init__(self,pos,vel,time,reference):
        Satellite.__init__(self,pos,vel,time,reference)

    def to_LVLH(self,pos_ref,LVLH):
        
        non_zero_pos = np.dot(LVLH,self.pos) #Position in LVLH, origin at centre of Earth
        pos = non_zero_pos - np.dot(LVLH,pos_ref) #Position, origin at chief spacecraft
        
        omega_L = np.array([0,0,self.reference.ang_vel]) #Angular momentum vector in LVLH frame
        vel = np.dot(LVLH,self.vel) - np.cross(omega_L,non_zero_pos) #Velocity, including rotating frame
        
        return LVLH_Sat(pos,vel,self.time,self.reference)

    def to_UV(self):
        pos = np.dot(self.reference.UVmat,self.pos)
        vel = np.dot(self.reference.UVmat,self.vel)
        return UV_Sat(pos,vel,self.time,self.reference)
        
        

class UV_Sat(Satellite):
    def __init__(self,pos,vel,time,reference):
        Satellite.__init__(self,pos,vel,time,reference)

    def to_ECI(self):
        inv_UV = np.linalg.inv(self.reference.UVmat)
        pos = np.dot(inv_UV,self.pos)
        vel = np.dot(inv_UV,self.vel)
        return ECI_Sat(pos,vel,self.time,self.reference)
        
        

class LVLH_Sat(Satellite):
    def __init__(self,pos,vel,time,reference):
        Satellite.__init__(self,pos,vel,time,reference)

    """ Takes a given state vector in LVLH coordinates and converts to ECI """
    """ Requires chief state and the change of basis matrix """
    def to_ECI(self,pos_ref,LVLH):
        
        inv_rotmat = LVLH.transpose() #LVLH to ECI change of basis matrix
        
        pos = np.dot(inv_rotmat,self.pos) + pos_ref #ECI position
        
        omega_L = np.array([0,0,self.reference.ang_vel]) #Angular momentum vector
        #Velocity in ECI frame, removing the rotation of the LVLH frame
        vel = np.dot(inv_rotmat,(self.vel + np.cross(omega_L,np.dot(LVLH,pos))))
        
        return ECI_Sat(pos,vel,self.time,self.reference)

    def to_Baseline(self,LVLH,Base):
        
        mat = np.dot(Base,LVLH.transpose())
        pos = np.dot(mat,self.pos)
        vel = np.dot(mat,self.vel)
        
        return Baseline_Sat(pos,vel,self.time,self.reference)


class Baseline_Sat(Satellite):
    def __init__(self,pos,vel,time,reference):
        Satellite.__init__(self,pos,vel,time,reference)

    def to_LVLH(self,LVLH,Base):
        
        mat = np.dot(LVLH,Base.transpose())
        
        pos = np.dot(mat,self.pos)
        vel = np.dot(mat,self.vel)
        s_hat = np.dot(mat,np.array([0,0,1]))
        
        return LVLH_Sat(pos,vel,self.time,self.reference)


def init_chief(reference,t,precession=False):
    
    pos_ref,vel_ref,LVLH,Base = reference.ref_orbit_pos(t,precession)

    return ECI_Sat(pos_ref,vel_ref,t,reference)


def init_deputy(reference,t,n,precession=False):

    if precession:
        #New coord system:
        z_hat = reference.h_0 #In direction of angular momentum

        x = reference.s_hat-z_hat*(np.dot(reference.s_hat,z_hat)) #Projection of the star vector on the orbital plane

        if (x == np.array([0.,0.,0.])).all():
            if (z_hat == np.array([1.,0.,0.])).all():
                x = np.array([0.,1.,0.])
            else:
                x = np.cross(z_hat,np.array([1.,0.,0.]))

        x_hat = x/np.linalg.norm(x)
        y_hat = np.cross(z_hat,x_hat) #Remaining orthogonal vector

        #Angle between angular momentum vector and star (checks are for precision errors):
        dot = np.dot(z_hat,reference.s_hat)
        if dot < -1.:
            dot = -1.
        elif dot > 1.:
            dot = 1.

        theta = np.arccos(dot)

        psi = reference.delta_r_max*np.cos(theta)/reference.R_orb #Angle between chief and deputy WRT Earth
        omega = -np.arctan(reference.delta_r_max/reference.R_orb*np.sin(theta)) #Amount of rotation

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
            q = reference.q1
        elif n == 2:
            q = reference.q2
        else:
            raise Exception("Bad Deputy number")

    pos_ref,vel_ref,LVLH,Base = reference.ref_orbit_pos(t,precession)

    return ECI_Sat(qt.rotate(pos_ref,q),qt.rotate(vel_ref,q),t,reference)
