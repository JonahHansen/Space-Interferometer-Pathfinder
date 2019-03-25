""" Perturbations Module """
import numpy as np
import astropy.constants as const
#import quaternions as qt
#from Schweighart_J2 import J2_pet

""" J2 Perturbation in ECI frame """
"""Input: positions of chief and deputy in ECI frame"""
def J2_pert(r_d,r_c,R_orb):

    J2 = 0.00108263 #J2 Parameter
    [x,y,z] = r_d #Deputy position in ECI coordinates

    #Calculate J2 acceleration from the equation in ECI frame
    J2_fac1 = 3/2*J2*const.GM_earth.value*const.R_earth.value**2/R_orb**5
    J2_fac2_d = 5*z**2/R_orb**2
    J2_p_d = J2_fac1*np.array([x*(J2_fac2_d-1),y*(J2_fac2_d-1),z*(J2_fac2_d-3)])

    #Calculate J2 acceleration for chief satellite
    J2_fac2_c = 5*r_c[2]**2/R_orb**2
    J2_p_c = J2_fac1*np.array([r_c[0]*(J2_fac2_c-1),r_c[1]*(J2_fac2_c-1),r_c[2]*(J2_fac2_c-3)])

    #Separation acceleration
    J2_p = J2_p_d - J2_p_c

    print("J2p" + str(np.linalg.norm(J2_p)))
    return J2_p

"""Calculates whether a satellite is in the Earth's shadow"""
"""Input: satellite position (r) and sun position (r_s)"""
def shadow(r,r_s):
    mag_r = np.linalg.norm(r)
    mag_r_s = np.linalg.norm(r_s)
    theta = np.arccos(np.dot(r_s,r)/(mag_r*mag_r_s))
    theta1 = np.arccos(const.R_earth.value/mag_r)
    theta2 = np.arccos(const.R_earth.value/mag_r_s)
    return (theta1 + theta2) >= theta

""" Calculates solar radiation perturbations in ECI frame """
""" Inputs: deputy position (r_d), chief position (r_c) and """
""" radiation parameters: As = area facing sun, Cr = reflection coefficient """
""" m = mass of spacecraft """
def solar_pert(r_d,r_c,As_d,Cr_d,m_d,As_c,Cr_c,m_c):

    r_s = const.au.value*np.array([0,1,0]) #Sun vector

    P_sr = 4.56e-6 #Solar Radiation Pressure Constant
    shadow_c = shadow(r_c,r_s)
    shadow_d = shadow(r_d,r_s)

    direction = r_s/np.linalg.norm(r_s)

    solar_p_c = -(shadow_c*1)*P_sr*Cr_c*As_c/m_c*direction
    solar_p_d = -(shadow_d*1)*P_sr*Cr_d*As_d/m_d*direction

    solar_p = solar_p_d - solar_p_c
    return solar_p

""" Calculates drag perturbations in ECI frame """
""" Inputs: deputy state, chief stage and drag parameters: """
""" rho = atmospheric density, C_D = drag coefficient """
""" A = Surface area perpendicular to velocity, m = mass of spacecraft """
def drag_pert(state_d,state_c,rho,C_D_c,C_D_d,A_c,A_d,m_c,m_d):

    #Angular velocity of the Earth
    omega_E = np.array([0,0,7.292115e-5])

    #Relative velocities to the Earth
    v_rel_c = state_c[3:] - np.cross(omega_E,state_c[:3])
    v_rel_d = state_d[3:] - np.cross(omega_E,state_d[:3])

    mag_v_rel_c = np.linalg.norm(v_rel_c)
    mag_v_rel_d = np.linalg.norm(v_rel_d)

    #Calculate perturbation accelerations
    drag_p_c = -1/2*rho*mag_v_rel_c*(C_D_c*A_c/m_c)*v_rel_c
    drag_p_d = -1/2*rho*mag_v_rel_d*(C_D_d*A_d/m_d)*v_rel_d

    #Relative acceleration back in LVLH frame
    drag_p = drag_p_c - drag_p_d

    print("dragp" + str(np.linalg.norm(drag_p)))
    return drag_p


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

    #Chief and deputy states in ECI frame
    ECI_c = ECI.chief_state(t)
    rot_mat = ECI.to_LVLH_mat(ECI_c) #Matrix to convert into LVLH
    ECI_d = ECI.LVLH_to_ECI_state(ECI_c,rot_mat,np.append(r,v))

    """ J2 Acceleration """

    J2_p = J2_pert(ECI_d[:3],ECI_c[:3],ECI.R_orb)
    LVLH_J2_p = np.dot(rot_mat,J2_p)
    #LVLH_J2_p = 0 #Comment out to use J2

    """ Solar Radiation """

    As_c = 0.2*0.3
    As_d = 0.1*0.3
    m_c = 8
    m_d = 4
    Cr_c = 1.5
    Cr_d = 1.5

    solar_p = solar_pert(ECI_d[:3],ECI_c[:3],As_c,Cr_c,m_c,As_d,Cr_d,m_d)
    LVLH_solar_p = np.dot(rot_mat,solar_p)
    #solar_p =0

    """ Drag """

    rho = 5.215e-13 #500km
    #rho = 3.561e-15 #1000km
    C_D = 2.1

    drag_p = drag_pert(ECI_d,ECI_c,rho,C_D,C_D,As_c,As_d,m_c,m_d)
    LVLH_drag_p = np.dot(rot_mat,drag_p)
    #LVLH_drag_p = 0

    """ Putting it together """
    #HCW Equations
    K = np.diag(np.array([3*n**2,0,-(n**2)]))
    Gamma2 = n**2/ECI.R_orb*np.array([-3*r[0]**2 + 1.5*r[1]**2 + 1.5*r[2]**2, 3*r[0]*r[1], 3*r[0]*r[2]])
    a = -2*np.cross(omega,v) + np.matmul(K,r) + Gamma2 + LVLH_J2_p + LVLH_solar_p + LVLH_drag_p

    #Acceleration vector - analytical version (See Butcher 18)
    #a = -2*np.cross(omega,v) - np.cross(omega,np.cross(omega,rd)) - const.GM_earth.value*rd/np.linalg.norm(rd)**3 + J2_pet_LVLH

    #Second half of the differential vector
    dX3 = a[0]
    dX4 = a[1]
    dX5 = a[2]
    return np.array([dX0,dX1,dX2,dX3,dX4,dX5])
