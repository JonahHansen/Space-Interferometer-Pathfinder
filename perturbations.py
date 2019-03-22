""" Perturbations Module """
import numpy as np
import astropy.constants as const
#import quaternions as qt
#from Schweighart_J2 import J2_pet


def J2_pert(r,r_c,rot_mat,R_orb):

    J2 = 0.00108263 #J2 Parameter
    [x,y,z] = np.dot(np.linalg.inv(rot_mat),r) + r_c #Deputy position in ECI coordinates

    #Calculate J2 acceleration from the equation in ECI frame
    J2_fac1 = 3/2*J2*const.GM_earth.value*const.R_earth.value**2/R_orb**5
    J2_fac2_dep = 5*z**2/R_orb**2
    J2_pert_dep = J2_fac1*np.array([x*(J2_fac2_dep-1),y*(J2_fac2_dep-1),z*(J2_fac2_dep-3)])

    #Calculate J2 acceleration for chief satellite
    J2_fac2_c = 5*r_c[2]**2/R_orb**2
    J2_pert_c = J2_fac1*np.array([r_c[0]*(J2_fac2_c-1),r_c[1]*(J2_fac2_c-1),r_c[2]*(J2_fac2_c-3)])

    #Separation acceleration
    J2_pert = J2_pert_dep - J2_pert_c

    #Convert back to LVLH frame
    J2_pert_LVLH = np.dot(rot_mat,J2_pert)
    return J2_pert_LVLH
    
def shadow(r,r_s):
    mag_r = np.linalg.norm(r)
    mag_r_s = np.linalg.norm(r_s)
    theta = np.arccos(np.dot(r_s,r)/(mag_r*mag_r_s))
    theta1 = np.arccos(const.R_earth.value/mag_r)
    theta2 = np.arccos(const.R_earth.value/mag_r_s)
    return (theta1 + theta2) >= theta

def solar_radiation(r,r_c,rot_mat,As_c,Cr_c,m_c,As_d,Cr_d,m_d):

    r_s = const.au.value*np.array([0,1,0])
    r_d = np.dot(np.linalg.inv(rot_mat),r) + r_c[:3]
    
    P_sr = 4.56e-6 #Solar Radiation Pressure Constant
    shadow_c = shadow(r_c,r_s)
    shadow_d = shadow(r_d,r_s)
    
    direction = r_s/np.linalg.norm(r_s)
    
    a_c = -(shadow_c*1)*P_sr*Cr_c*As_c/m_c*direction
    a_d = -(shadow_d*1)*P_sr*Cr_d*As_d/m_d*direction
    
    #print(a_c,a_d)

    pert = np.dot(rot_mat,(a_d-a_c))
    return pert
    
def drag_pert(r,v,rot_mat,omega_F,rho,C_D,A,m):
    omega_E = np.dot(rot_mat,np.array([0,0,7.292115e-5])) - omega_F
    v_rel = v - np.cross(omega_E,r)
    mag_v_rel = np.linalg.norm(v_rel)
    pert = -1/2*rho*mag_v_rel*(C_D*A/m)*v_rel
    return pert


""" Differential equation function"""
def dX_dt(t, state, ECI):
    r = state[:3] #Position
    v = state[3:] #Velocity

    #First half of the differential vector
    dX0 = v[0]
    dX1 = v[1]
    dX2 = v[2]

    #Position in LVLH frame, with origin at centre of the Earth
    rd = np.array([ECI.R_orb+r[0],r[1],r[2]])

    n = ECI.ang_vel #Angular velocity
    omega = np.array([0,0,n]) #Angular velocity vector in LVLH frame

    r_c = ECI.chief_state(t)[:3] #Chief position in ECI at time t
    rot_mat = ECI.to_LVLH_mat(r_c) #Matrix to convert into LVLH

    """ J2 Acceleration """

    J2_p = J2_pert(r,r_c,rot_mat,ECI.R_orb)
    J2_p = 0 #Comment out to use J2

    """ Solar Radiation """
    """
    As_c = 0.2*0.3
    As_d = 0.1*0.3
    m_c = 8
    m_d = 4
    Cr_c = 1.5
    Cr_d = 1.5
    
    solar_p = solar_radiation(r,r_c,rot_mat,As_c,Cr_c,m_c,As_d,Cr_d,m_d)
    """
    """ Drag """
    """
    rho = 5.215e-13 #500km, 3.561e-15 for 1000km
    C_D = 2.1
    A = As_d

    drag_p = drag_pert(r,v,rot_mat,omega,rho,C_D,A,m_d)
    """
    #print(J2_p,solar_p,drag_p)

    #HCW Equations
    K = np.diag(np.array([3*n**2,0,-(n**2)]))
    Gamma2 = n**2/ECI.R_orb*np.array([-3*r[0]**2 + 1.5*r[1]**2 + 1.5*r[2]**2, 3*r[0]*r[1], 3*r[0]*r[2]])
    a = -2*np.cross(omega,v) + np.matmul(K,r) + Gamma2 + J2_p # solar_p + drag_p

    #Acceleration vector - analytical version (See Butcher 18)
    #a = -2*np.cross(omega,v) - np.cross(omega,np.cross(omega,rd)) - const.GM_earth.value*rd/np.linalg.norm(rd)**3 + J2_pet_LVLH

    #Second half of the differential vector
    dX3 = a[0]
    dX4 = a[1]
    dX5 = a[2]
    return np.array([dX0,dX1,dX2,dX3,dX4,dX5])