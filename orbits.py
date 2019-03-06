import numpy as np
import astropy.constants as const
import quaternions as qt

class sat_orbit:
    def __init__(self, n_p, R_orb):

        self.n_p = n_p
        self.phase = np.linspace(0, 2*np.pi, n_p)

        self.R_orb = R_orb
        self.period = 2*np.pi*np.sqrt((R_orb)**3/const.GM_earth).value #In seconds.
        self.ang_vel = 2*np.pi/self.period

        self.chief_pos = np.zeros((n_p,3))
        self.chief_vel = np.zeros((n_p,3))
        self.deputy1_pos = np.zeros((n_p,3))
        self.deputy1_vel = np.zeros((n_p,3))
        self.deputy2_pos = np.zeros((n_p,3))
        self.deputy2_vel = np.zeros((n_p,3))

        self.deputy1_pos_sep = np.zeros((n_p,3))
        self.deputy1_vel_sep = np.zeros((n_p,3))
        self.deputy2_pos_sep = np.zeros((n_p,3))
        self.deputy2_vel_sep = np.zeros((n_p,3))

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

class ECEF_orbit(sat_orbit):

    def __init__(self, n_p, R_orb, delta_r_max, inc_0, Om_0, ra, dec):
        sat_orbit.__init__(self, n_p, R_orb)

        self.Om_0 = Om_0
        self.inc_0 = inc_0

        self.delta_r_max = delta_r_max

        self.s_hat = [np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)]

        for i in range(self.n_p):
            self.chief_pos[i,0] = np.cos(self.phase[i]) * self.R_orb
            self.chief_pos[i,1] = np.sin(self.phase[i]) * self.R_orb
            self.chief_vel[i,0] = -np.sin(self.phase[i]) * self.R_orb * self.ang_vel
            self.chief_vel[i,1] = np.cos(self.phase[i]) * self.R_orb * self.ang_vel

        #Initial axis unit vectors
        xaxis = np.array([1,0,0])
        yaxis = np.array([0,1,0])
        zaxis = np.array([0,0,1])

        #Quaternion rotation, using Mike's "No phase difference" rotation!
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
        y = self.s_hat-z_hat*(np.dot(self.s_hat,z_hat)) #Projection of the star vector on the orbital plane
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

        #Rotate the chiefs orbit
        self.deputy1_pos = qt.rotate_points(self.chief_pos,self.q1)
        self.deputy1_vel = qt.rotate_points(self.chief_vel,self.q1)
        self.deputy2_pos = qt.rotate_points(self.chief_pos,self.q2)
        self.deputy2_vel = qt.rotate_points(self.chief_vel,self.q2)

        self.deputy1_pos_sep = (self.deputy1_pos - self.chief_pos)
        self.deputy2_pos_sep = (self.deputy2_pos - self.chief_pos)
        self.deputy1_vel_sep = (self.deputy1_vel - self.chief_vel)
        self.deputy2_vel_sep = (self.deputy2_vel - self.chief_vel)

class LVLH_orbit(sat_orbit):

    def __init__(self, n_p, R_orb, ECEF):

        sat_orbit.__init__(self, n_p, R_orb)
        self.s_hats = np.zeros((n_p,3))

        for ix in range(ECEF.n_p):

            h_hat = qt.rotate(np.array([0,0,1]),ECEF.q0) #Angular momentum vector (rotated "z" axis)
            r_hat = ECEF.chief_pos[ix]/np.linalg.norm(ECEF.chief_pos[ix]) #Position vector pointing away from the centre of the Earth
            v_hat = np.cross(h_hat,r_hat) #Velocity vector pointing counter-clockwise

            rot_mat = np.array([r_hat,v_hat,h_hat]) #Rotation matrix from three unit vectors

            self.chief_pos[ix] = np.dot(rot_mat,ECEF.chief_pos[ix])-np.dot(rot_mat,ECEF.chief_pos[ix])
            self.chief_vel[ix] = np.dot(rot_mat,ECEF.chief_vel[ix])
            self.deputy1_pos[ix] = np.dot(rot_mat,ECEF.deputy1_pos[ix])-np.dot(rot_mat,ECEF.chief_pos[ix])
            self.deputy1_vel[ix] = np.dot(rot_mat,ECEF.deputy1_vel[ix])
            self.deputy2_pos[ix] = np.dot(rot_mat,ECEF.deputy2_pos[ix])-np.dot(rot_mat,ECEF.chief_pos[ix])
            self.deputy2_vel[ix] = np.dot(rot_mat,ECEF.deputy2_vel[ix])
            self.s_hats[ix] = np.dot(rot_mat,ECEF.s_hat)
            
        self.deputy1_pos_sep = (self.deputy1_pos - self.chief_pos)
        self.deputy2_pos_sep = (self.deputy2_pos - self.chief_pos)
        self.deputy1_vel_sep = (self.deputy1_vel - self.chief_vel)
        self.deputy2_vel_sep = (self.deputy2_vel - self.chief_vel)

class Baseline_orbit(sat_orbit):

    def __init__(self, n_p, R_orb, ECEF):

        sat_orbit.__init__(self, n_p, R_orb)

        for ix in range(ECEF.n_p):
            
            b_hat = (ECEF.deputy2_pos[ix] - ECEF.chief_pos[ix])/np.linalg.norm(ECEF.deputy2_pos[ix] - ECEF.chief_pos[ix]) #Direction along baseline
            k_hat = np.cross(ECEF.s_hat,b_hat) #Other direction

            rot_mat = np.array([k_hat,b_hat,ECEF.s_hat]) #Create rotation matrix

            
            self.chief_pos[ix] = np.dot(rot_mat,ECEF.chief_pos[ix])-np.dot(rot_mat,ECEF.chief_pos[ix])
            self.chief_vel[ix] = np.dot(rot_mat,ECEF.chief_vel[ix])
            self.deputy1_pos[ix] = np.dot(rot_mat,ECEF.deputy1_pos[ix])-np.dot(rot_mat,ECEF.chief_pos[ix])
            self.deputy1_vel[ix] = np.dot(rot_mat,ECEF.deputy1_vel[ix])
            self.deputy2_pos[ix] = np.dot(rot_mat,ECEF.deputy2_pos[ix])-np.dot(rot_mat,ECEF.chief_pos[ix])
            self.deputy2_vel[ix] = np.dot(rot_mat,ECEF.deputy2_vel[ix])

        self.deputy1_pos_sep = (self.deputy1_pos - self.chief_pos)
        self.deputy2_pos_sep = (self.deputy2_pos - self.chief_pos)
        self.deputy1_vel_sep = (self.deputy1_vel - self.chief_vel)
        self.deputy2_vel_sep = (self.deputy2_vel - self.chief_vel)