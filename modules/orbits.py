"""ORBIT and SATELLITE Classes - Calculates orbits and converts between orbital frames"""

import numpy as np
import astropy.constants as const
import modules.quaternions as qt

"""
Reference Orbit class: Use this to first calculate the orbit from scratch.
Initially in ECI frame
Inputs:
    n_p - number of phases
    R_orb - radius of orbit
    delta_r_max - maximum separation of deputies from chief
    inc_0,Om_0 - orientation of chief orbit
    ra,dec - position of star to be observed
"""
class Reference_orbit:

    def __init__(self, R_orb, delta_r_max, inc_0, Om_0, ra, dec):

        self.Om_0 = Om_0 #Longitude of the ascending node
        self.inc_0 = inc_0 #Inclination

        self.delta_r_max = delta_r_max #Maximum separation between chief and deputy

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
           self.u_hat = np.cross(self.s_hat,zaxis)/np.linalg.norm(np.cross(self.s_hat,zaxis))
        self.v_hat = np.cross(self.s_hat,self.u_hat)/np.linalg.norm(np.cross(self.s_hat,self.u_hat))

        #J2 Perturbation constants and angular velocities
        J2 = 0.00108263
        self.Sch_s = 3/8*const.R_earth.value**2/R_orb**2*J2*(1+3*np.cos(2*inc_0))
        self.Sch_c = np.sqrt(1+self.Sch_s)
        self.Sch_k = self.ang_vel*self.Sch_c + 3*self.ang_vel*J2*const.R_earth.value**2/(2*R_orb**2)*(np.cos(inc_0)**2)

        #Schweighart periods?
        self.periodK = 2*np.pi/self.Sch_k
        self.periodNC = 2*np.pi/(self.Sch_c*self.ang_vel)

        #Quaternion rotation of reference (chief) orbit
        q_Om = qt.to_q(zaxis,Om_0)
        q_inc = qt.to_q(xaxis,inc_0)
        #self.q0 = qt.comb_rot(qt.comb_rot(qt.conjugate(q_Om),q_inc),q_Om)
        self.q0 = qt.comb_rot(q_inc,q_Om)

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


    """
    Find u and v vectors given the deputy state vectors
    Inputs:
        dep1 - Deputy 1
        dep2 - Deputy 2
    Outputs:
        Point in UV plane
    """
    def uv(self,dep1,dep2):
        sep = dep2.pos - dep1.pos #Baseline vector
        u = np.dot(sep,self.u_hat)
        v = np.dot(sep,self.v_hat)
        return np.array([u,v])


    """
    Returns the position and velocity of a point on the reference orbit at a given time t
    Also returns the ECI to LVLH/Baseline change of basis matrices at this point
    Inputs:
        t - time to evaluate the position
        precession - are we considering J2?
    Outputs:
        pos - position of the reference point
        vel - velocity of the reference point
        LVLH_mat - matrix to change from ECI to LVLH based on this point
        Base_mat - matrix to change from ECI to Baseline based on this point
    """
    def ref_orbit_pos(self,t,precession=True):
        
        #If not precession, set J2 to 0
        if precession:
            J2 = 0.00108263
            omega = self.Sch_k
        else:
            J2 = 0
            omega = self.ang_vel

        sq_mu = np.sqrt(const.GM_earth.value)
        R_e = const.R_earth.value

        #Parameters from Schweighart
        factor = 3*sq_mu*J2*R_e**2/(2*self.R_orb**(3.5))*np.cos(self.inc_0)

        #Reference orbit parameters
        i = self.inc_0 - factor/omega*np.sin(self.inc_0)
        Om = self.Om_0 - factor*t
        th = omega*t
        dot_i = 0
        dot_Om = -factor
        dot_th = omega

        pos = np.zeros(3)
        vel = np.zeros(3)

        #Calculate state from given parameters
        pos[0] = self.R_orb*(np.cos(Om)*np.cos(th)-np.sin(Om)*np.sin(th)*np.cos(i))
        pos[1] = self.R_orb*(np.sin(Om)*np.cos(th)+np.cos(Om)*np.sin(th)*np.cos(i))
        pos[2] = self.R_orb*(np.sin(th)*np.sin(i))

        vel[0] = self.R_orb*(-np.sin(Om)*np.cos(th)*dot_Om - np.cos(Om)*np.sin(th)*dot_th -
                              np.cos(Om)*np.sin(th)*np.cos(i)*dot_Om - np.sin(Om)*np.cos(th)*np.cos(i)*dot_th)
        vel[1] = self.R_orb*(np.cos(Om)*np.cos(th)*dot_Om - np.sin(Om)*np.sin(th)*dot_th -
                             np.sin(Om)*np.sin(th)*np.cos(i)*dot_Om + np.cos(Om)*np.cos(th)*np.cos(i)*dot_th)
        vel[2] = self.R_orb*(np.cos(th)*np.sin(i)*dot_th)

        #LVLH matrix
        rho_hat = pos/np.linalg.norm(pos) #Position unit vector (rho)
        xi_hat = vel/np.linalg.norm(vel) #Velocity unit vector (xi)
        eta_hat = np.cross(rho_hat,xi_hat) #Angular momentum vector (eta)
        LVLH_mat = np.array([rho_hat,xi_hat,eta_hat]) #LVLH rotation matrix

        #Baseline matrix
        #Catch all fix if the satellites essentially crash into each other
        if not np.any(np.cross(rho_hat,self.s_hat)):
            pos_ref,vel_ref,LVLH,Base_pos = self.ref_orbit_pos(t+0.1,precession)
            pos_ref,vel_ref,LVLH,Base_neg = self.ref_orbit_pos(t-0.1,precession)
            b_hat_pos = Base_pos[0]
            b_hat_neg = Base_neg[0]
            b = (b_hat_pos + b_hat_neg)*0.5
            b_hat = b/np.linalg.norm(b)
        else:
            b_hat = np.cross(rho_hat,self.s_hat)/np.linalg.norm(np.cross(rho_hat,self.s_hat)) #Baseline unit vector
        o_hat = np.cross(self.s_hat,b_hat) #Other unit vector
        Base_mat = np.array([b_hat,o_hat,self.s_hat]) #Baseline rotation matrix

        return pos,vel,LVLH_mat,Base_mat
        

    """
    Returns the position and velocity of a given chief state (instead of reference orbit above)
    Also returns the ECI to LVLH/Baseline change of basis matrices for this chief state
    Inputs:
        state - chief state
        state2 - secondary state to calculate position of LVLH matrix with (only required if
                 the satellites are crashing into each other)
    Outputs:
        pos - position of the chief
        vel - velocity of the chief
        LVLH_mat - matrix to change from ECI to LVLH based on this state
        Base_mat - matrix to change from ECI to Baseline based on this state
    """
    def chief_orbit_pos(self,state,state2):
        if not np.any(state):
            raise Exception("You didn't give me the chief state. Check state != 0")
        pos = state[:3]
        vel = state[3:]

        #LVLH matrix
        rho_hat = pos/np.linalg.norm(pos) #Position unit vector (rho)
        xi_hat = vel/np.linalg.norm(vel) #Velocity unit vector (xi)
        eta_hat = np.cross(rho_hat,xi_hat)/np.linalg.norm(np.cross(rho_hat,xi_hat)) #Angular momentum vector (eta)
        LVLH_mat = np.array([rho_hat,xi_hat,eta_hat]) #LVLH rotation matrix

        #Catch all fix if the satellites essentially crash into each other
        if not np.any(np.cross(rho_hat,self.s_hat)):
            if not np.any(state2):
                raise Exception("Baseline not found. Check state2 != 0")
            #Use other state (state2) for the position if the first one is "bad"
            rho_hat = state2[:3]/np.linalg.norm(state2[:3])

        #Baseline matrix
        b_hat = np.cross(rho_hat,self.s_hat)/np.linalg.norm(np.cross(rho_hat,self.s_hat)) #Baseline unit vector
        o_hat = np.cross(self.s_hat,b_hat) #Other unit vector
        Base_mat = np.array([b_hat,o_hat,self.s_hat]) #Baseline rotation matrix

        return pos,vel,LVLH_mat,Base_mat


#################################################################################################################


""" Satellite class: Base class for the other reference frames
    Inputs:
        pos - position of satellite
        vel - velocity of satellite
        time - time of position/velocity
        reference - reference orbit
"""
class Satellite:
    def __init__(self,pos,vel,time,reference):
        self.pos = pos #position
        self.vel = vel #velocity
        self.time = time
        self.state = np.concatenate((self.pos,self.vel)) #state vector
        self.reference = reference


    """ Orbital orientation elements (i,Om) from state vector """
    def orbit_elems(self):
        h = np.cross(self.pos,self.vel) #Angular momentum vector
        n = np.cross(np.array([0,0,1]),h) #Nodal vector
        i = np.arccos(h[2]/np.linalg.norm(h)) #Inclination
        Omega = np.arccos(n[0]/np.linalg.norm(n)) #Longitude of the ascending node
        if n[1] < 0:
            Omega = 360 - Omega
        return i,Omega



""" Satellite in the ECI (Inertial frame) """
class ECI_Sat(Satellite):
    def __init__(self,pos,vel,time,reference):
        Satellite.__init__(self,pos,vel,time,reference)


    """ Change to LVLH frame.
        Inputs:
            precession - Are we accounting for precession due to J2?
            ref_orbit - Are we using the reference orbit?
            state - state of the chief satellite (not needed if ref_orbit is True)
            state2 - slightly perturbed chief state to fix issue when satellites crash (see chief_orbit_pos).
                     Not strictly needed.
    """
    def to_LVLH(self,precession=True,ref_orbit=False,state=np.zeros(6),state2=np.zeros(6)):
        
        #Get the position/velocity and rotation matrix from either the reference orbit, or a given chief state
        if ref_orbit:
            pos_ref,vel_ref,LVLH,Base = self.reference.ref_orbit_pos(self.time,precession)
        else:
            pos_ref,vel_ref,LVLH,Base = self.reference.chief_orbit_pos(state,state2)

        #Position in LVLH, origin at centre of Earth
        non_zero_pos = np.dot(LVLH,self.pos)
        #Position, origin at chief spacecraft
        pos = non_zero_pos - np.dot(LVLH,pos_ref)
        #Angular velocity vector
        omega_L = np.array([0,0,np.linalg.norm(np.cross(pos_ref,vel_ref)/np.linalg.norm(pos_ref)**2)])
        #Velocity, including rotating frame
        vel = np.dot(LVLH,self.vel) - np.cross(omega_L,non_zero_pos)
        
        return LVLH_Sat(pos,vel,self.time,self.reference)


    """ Change to Baseline frame via LVLH.
        Inputs:
            precession - Are we accounting for precession due to J2?
            ref_orbit - Are we using the reference orbit?
            state - state of the chief satellite (not needed if ref_orbit is True)
            state2 - slightly perturbed chief state to fix issue when satellites crash (see chief_orbit_pos).
                     Not strictly needed.
    """
    def to_Baseline(self,precession=True,ref_orbit=False,state=np.zeros(6),state2=np.zeros(6)):
        return self.to_LVLH(precession,ref_orbit,state,state2).to_Baseline(precession,ref_orbit,state,state2)
        
        
    """ Change to curvilinear LVLH frame via rectangular LVLH.
        Inputs:
            precession - Are we accounting for precession due to J2?
            ref_orbit - Are we using the reference orbit?
            state - state of the chief satellite (not needed if ref_orbit is True)
            state2 - slightly perturbed chief state to fix issue when satellites crash (see chief_orbit_pos).
                     Not strictly needed.
    """
    def to_Curvy(self,precession=True,ref_orbit=False,state=np.zeros(6),state2=np.zeros(6)):
        return self.to_LVLH(precession,ref_orbit,state,state2).to_Curvy()



""" Satellite in rectangular LVLH frame """
class LVLH_Sat(Satellite):
    def __init__(self,pos,vel,time,reference):
        Satellite.__init__(self,pos,vel,time,reference)


    """ Change to ECI frame.
        Inputs:
            precession - Are we accounting for precession due to J2?
            ref_orbit - Are we using the reference orbit?
            state - state of the chief satellite (not needed if ref_orbit is True)
            state2 - slightly perturbed chief state to fix issue when satellites crash (see chief_orbit_pos).
                     Not strictly needed.
    """
    def to_ECI(self,precession=True,ref_orbit=False,state=np.zeros(6),state2=np.zeros(6)):
        
        #Get the position/velocity and rotation matrix from either the reference orbit, or a given chief state
        if ref_orbit:
            pos_ref,vel_ref,LVLH,Base = self.reference.ref_orbit_pos(self.time,precession)
        else:
            pos_ref,vel_ref,LVLH,Base = self.reference.chief_orbit_pos(state,state2)

        #LVLH to ECI change of basis matrix
        inv_rotmat = np.linalg.inv(LVLH)
        #ECI position
        pos = np.dot(inv_rotmat,self.pos) + pos_ref
        #Angular velocity vector
        omega_L = np.array([0,0,np.linalg.norm(np.cross(pos_ref,vel_ref)/np.linalg.norm(pos_ref)**2)])
        #Velocity in ECI frame, removing the rotation of the LVLH frame
        vel = np.dot(inv_rotmat,(self.vel + np.cross(omega_L,np.dot(LVLH,pos))))
        
        return ECI_Sat(pos,vel,self.time,self.reference)


    """ Change to Baseline frame.
        Inputs:
            precession - Are we accounting for precession due to J2?
            ref_orbit - Are we using the reference orbit?
            state - state of the chief satellite (not needed if ref_orbit is True)
            state2 - slightly perturbed chief state to fix issue when satellites crash (see chief_orbit_pos).
                     Not strictly needed.
    """
    def to_Baseline(self,precession=True,ref_orbit=False,state=np.zeros(6),state2=np.zeros(6)):
        
        #Get the position/velocity and rotation matrix from either the reference orbit, or a given chief state
        if ref_orbit:
            pos_ref,vel_ref,LVLH,Base = self.reference.ref_orbit_pos(self.time,precession)
        else:
            pos_ref,vel_ref,LVLH,Base = self.reference.chief_orbit_pos(state,state2)

        #Change in basis is simply a rotation (change via the ECI frame)
        mat = np.dot(Base,np.linalg.inv(LVLH))
        pos = np.dot(mat,self.pos)
        vel = np.dot(mat,self.vel)
        
        return Baseline_Sat(pos,vel,self.time,self.reference)


    """ Change to Curvilinear LVLH frame. See Thesis for more information
        Inputs:
            precession - Are we accounting for precession due to J2?
            ref_orbit - Are we using the reference orbit?
            state - state of the chief satellite (not needed if ref_orbit is True)
            state2 - slightly perturbed chief state to fix issue when satellites crash (see chief_orbit_pos).
                     Not strictly needed.
    """
    def to_Curvy(self,precession=True,ref_orbit=False,state=np.zeros(6),state2=np.zeros(6)):

        #Position and velocity in rectangular LVLH frame
        x,y,z = self.pos
        dx,dy,dz = self.vel
        
        #Add in the orbital radius
        R0 = self.reference.R_orb
        R0x = R0 + x
        new_r = np.array([R0,0,0]) + self.pos
        Rd = np.linalg.norm(new_r)
        
        #Polar and azimuthal angles
        phi = np.arctan(y/R0x)
        theta = np.arcsin(z/Rd)
        
        #Derivatives
        dRd = np.dot(new_r,self.vel)/Rd
        dphi = (dy*R0x - y*dx)/(y**2+R0x**2)
        dtheta = (dz*Rd-z*dRd)/np.sqrt(Rd**4-z**2*Rd**2)

        #Curvy position and velocity
        pos = np.array([Rd-R0,R0*phi,R0*theta])
        vel = np.array([dRd,R0*dphi,R0*dtheta])
        
        return Curvy_Sat(pos,vel,self.time,self.reference)



""" Satellite in Baseline frame """
class Baseline_Sat(Satellite):
    def __init__(self,pos,vel,time,reference):
        Satellite.__init__(self,pos,vel,time,reference)


    """ Change to LVLH frame.
        Inputs:
            precession - Are we accounting for precession due to J2?
            ref_orbit - Are we using the reference orbit?
            state - state of the chief satellite (not needed if ref_orbit is True)
            state2 - slightly perturbed chief state to fix issue when satellites crash (see chief_orbit_pos).
                     Not strictly needed.
    """
    def to_LVLH(self,precession=True,ref_orbit=False,state=np.zeros(6),state2=np.zeros(6)):
        
        #Get the position/velocity and rotation matrix from either the reference orbit, or a given chief state
        if ref_orbit:
            pos_ref,vel_ref,LVLH,Base = self.reference.ref_orbit_pos(self.time,precession)
        else:
            pos_ref,vel_ref,LVLH,Base = self.reference.chief_orbit_pos(state,state2)

        #Change in basis is simply a rotation (change via the ECI frame)
        mat = np.dot(LVLH,np.linalg.inv(Base))
        pos = np.dot(mat,self.pos)
        vel = np.dot(mat,self.vel)
        
        return LVLH_Sat(pos,vel,self.time,self.reference)


    """ Change to curvilinear LVLH frame via rectangular LVLH frame.
        Inputs:
            precession - Are we accounting for precession due to J2?
            ref_orbit - Are we using the reference orbit?
            state - state of the chief satellite (not needed if ref_orbit is True)
            state2 - slightly perturbed chief state to fix issue when satellites crash (see chief_orbit_pos).
                     Not strictly needed.
    """
    def to_Curvy(self,precession=True,ref_orbit=False,state=np.zeros(6),state2=np.zeros(6)):
        return self.to_LVLH(precession,ref_orbit,state,state2).to_Curvy()


    """ Change to ECI frame via LVLH frame.
        Inputs:
            precession - Are we accounting for precession due to J2?
            ref_orbit - Are we using the reference orbit?
            state - state of the chief satellite (not needed if ref_orbit is True)
            state2 - slightly perturbed chief state to fix issue when satellites crash (see chief_orbit_pos).
                     Not strictly needed.
    """
    def to_ECI(self,precession=True,ref_orbit=False,state=np.zeros(6),state2=np.zeros(6)):
        return self.to_LVLH(precession,ref_orbit,state,state2).to_ECI(precession,ref_orbit,state,state2)



""" Satellite in Curvilinear LVLH frame """
class Curvy_Sat(Satellite):
    def __init__(self,pos,vel,time,reference):
        Satellite.__init__(self,pos,vel,time,reference)


    """ Change to rectangular LVLH frame.
        Inputs:
            precession - Are we accounting for precession due to J2?
            ref_orbit - Are we using the reference orbit?
            state - state of the chief satellite (not needed if ref_orbit is True)
            state2 - slightly perturbed chief state to fix issue when satellites crash (see chief_orbit_pos).
                     Not strictly needed.
    """
    def to_LVLH(self,precession=True,ref_orbit=False,state=np.zeros(6),state2=np.zeros(6)):

        #Curvy positions and velocities
        xc,yc,zc = self.pos
        dxc,dyc,dzc = self.vel
        R0 = self.reference.R_orb

        #Define length from centre of Earth
        Rd = R0 + xc
        
        #Polar and azimuthal angles
        phi = yc/R0
        theta = zc/R0

        #New positions
        x = Rd*np.cos(phi)*np.cos(theta) - R0
        y = Rd*np.sin(phi)*np.cos(theta)
        z = Rd*np.sin(theta)

        #Angular velocities
        dphi = dyc/R0
        dtheta = dzc/R0

        #New velocities
        dx = dxc*np.cos(phi)*np.cos(theta) - Rd*np.sin(phi)*np.cos(theta)*dphi - Rd*np.cos(phi)*np.sin(theta)*dtheta
        dy = dxc*np.sin(phi)*np.cos(theta) + Rd*np.cos(phi)*np.cos(theta)*dphi - Rd*np.sin(phi)*np.sin(theta)*dtheta
        dz = dxc*np.sin(theta) + Rd*np.cos(theta)*dtheta
    
        return LVLH_Sat(np.array([x,y,z]),np.array([dx,dy,dz]),self.time,self.reference)


    """ Change to Baseline frame via rectangular LVLH frame.
        Inputs:
            precession - Are we accounting for precession due to J2?
            ref_orbit - Are we using the reference orbit?
            state - state of the chief satellite (not needed if ref_orbit is True)
            state2 - slightly perturbed chief state to fix issue when satellites crash (see chief_orbit_pos).
                     Not strictly needed.
    """
    def to_Baseline(self,precession=True,ref_orbit=False,state=np.zeros(6),state2=np.zeros(6)):
        return self.to_LVLH().to_Baseline(precession,ref_orbit,state,state2)
        
        
    """ Change to ECI frame via rectangular LVLH frame.
        Inputs:
            precession - Are we accounting for precession due to J2?
            ref_orbit - Are we using the reference orbit?
            state - state of the chief satellite (not needed if ref_orbit is True)
            state2 - slightly perturbed chief state to fix issue when satellites crash (see chief_orbit_pos).
                     Not strictly needed.
    """
    def to_ECI(self,precession=True,ref_orbit=False,state=np.zeros(6),state2=np.zeros(6)):
        return self.to_LVLH().to_ECI(precession,ref_orbit,state,state2)


###############################################################################################################


"""
Initialise the chief satellite at t = 0 from the reference orbit
Inputs:
    reference - reference orbit
    precession - Are we accounting for precession due to J2?
    time - time at which to retrive the state
Outputs:
    ECI state of the chief satellite at time "time"
"""
def init_chief(reference,precession=True,time=0):
    pos_ref,vel_ref,LVLH,Base = reference.ref_orbit_pos(time,precession)
    return ECI_Sat(pos_ref,vel_ref,time,reference)



""" Initialise a deputy at t=0 from the reference orbit or a given chief position.
Inputs:
    reference - reference orbit
    n - number of the deputy (1 or 2)
    precession - Are we accounting for precession due to J2?
    time - time at which to retrive the state
    ref_orbit - Are we using the reference orbit?
    state - state of the chief satellite (not needed if ref_orbit is True)
    state2 - slightly perturbed chief state to fix issue when satellites crash (see chief_orbit_pos).
             Not strictly needed.
Outputs:
    ECI state of the deputy satellite at time "time"
"""
def init_deputy(reference,n,precession=True,time=0,ref_orbit=True,state=np.zeros(6),state2=np.zeros(6)):

    #Get the position/velocity and rotation matrix from either the reference orbit, or a given chief state
    if ref_orbit:
        pos_ref,vel_ref,LVLH,Base = reference.ref_orbit_pos(time,precession)
    else:
        pos_ref,vel_ref,LVLH,Base = reference.chief_orbit_pos(state,state2)

    #If precession, recalculate the quaternion to define a new rotation for the precessed positions/velocities
    if precession:
        #New coord system:
        z_hat = np.cross(pos_ref,vel_ref)/np.linalg.norm(np.cross(pos_ref,vel_ref)) #In direction of angular momentum

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

        #print(psi,omega,vel_ref)

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
        #Init satellite using the quaternion defined in the reference orbit
        if n == 1:
            q = reference.q1
        elif n == 2:
            q = reference.q2
        else:
            raise Exception("Bad Deputy number")

    #Rotate the reference/chief position using the quaternions
    return ECI_Sat(qt.rotate(pos_ref,q),qt.rotate(vel_ref,q),time,reference)
