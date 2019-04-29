""" Perturbations Module """
import numpy as np
import astropy.constants as const

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

    return drag_p


""" Master Differential equation function for the integrator - Integrates HCW equations """
""" Takes in a time and a state vector, as well as the reference orbit """
""" and list of required perturbations. """
""" Returns the derivative """
def dX_dt(t, state, ECI, perturbations_ls):
    r = state[:3] #Position
    v = state[3:] #Velocity

    #First half of the differential vector (derivative of position, velocity)
    dX0 = v[0]
    dX1 = v[1]
    dX2 = v[2]

    n = ECI.ang_vel #Angular velocity
    omega = np.array([0,0,n]) #Angular velocity vector in LVLH frame

    #Calculate Chief and deputy states in ECI frame at the time t
    ECI_c = ECI.chief_state(t)
    rot_mat = ECI.to_LVLH_mat(ECI_c) #Matrix to convert into LVLH
    ECI_d = ECI.LVLH_to_ECI_state(ECI_c,rot_mat,np.append(r,v))

    """ J2 Acceleration """

    if 1 in perturbations_ls:
        J2_p = J2_pert(ECI_d[:3],ECI_c[:3],ECI.R_orb) #Calculate J2 in ECI frame
        LVLH_J2_p = np.dot(rot_mat,J2_p) #Convert to LVLH frame
    else:
        LVLH_J2_p = 0

    """ Solar Radiation """

    if 2 in perturbations_ls:
        As_c = 0.1*0.3
        As_d = 0.1*0.3
        m_c = 8
        m_d = 4
        Cr_c = 1.5
        Cr_d = 1.5

        solar_p = solar_pert(ECI_d[:3],ECI_c[:3],As_c,Cr_c,m_c,As_d,Cr_d,m_d) #In ECI frame
        LVLH_solar_p = np.dot(rot_mat,solar_p) #To LVLH frame
    else:
        LVLH_solar_p =0

    """ Drag """

    if 3 in perturbations_ls:
        rho = 7.85e-13 #500km COSPAR CIRA-2012
        #rho = 6.59e-15 #1000km
        C_D = 2.1
        As_c = 0.1*0.3
        As_d = 0.1*0.3
        m_c = 8
        m_d = 4

        drag_p = drag_pert(ECI_d,ECI_c,rho,C_D,C_D,As_c,As_d,m_c,m_d) #In ECI frame
        LVLH_drag_p = np.dot(rot_mat,drag_p) #To LVLH frame
    else:
        LVLH_drag_p = 0

    """ Putting it together """

    #HCW Equations (second order correction, see Butcher 16)
    K = np.diag(np.array([3*n**2,0,-(n**2)]))
    Gamma2 = n**2/ECI.R_orb*np.array([-3*r[0]**2 + 1.5*r[1]**2 + 1.5*r[2]**2, 3*r[0]*r[1], 3*r[0]*r[2]])
    #Gamma2 = 0

    #Acceleration is the HCW Equations, plus the required perturbations
    a = -2*np.cross(omega,v) + np.matmul(K,r) + Gamma2 + LVLH_J2_p + LVLH_solar_p + LVLH_drag_p

    #Print kinetic energy while integrating
    print(-np.linalg.norm(v)**2/2)

    #Second half of the differential vector (derivative of velocity, acceleration)
    dX3 = a[0]
    dX4 = a[1]
    dX5 = a[2]
    return np.array([dX0,dX1,dX2,dX3,dX4,dX5])
