""" Perturbations Module """
import numpy as np
import astropy.constants as const
import modules.orbits as orbits

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

""" Master Differential equation function for the integrator - Integrates HCW equations """
""" Takes in a time and a state vector, as well as the reference orbit """
""" and list of required perturbations. """
""" Returns the derivative """
def dX_dt(t, state, ref):
    r = state[:3] #Position
    v = state[3:] #Velocity

    #First half of the differential vector (derivative of position, velocity)
    dX0 = v[0]
    dX1 = v[1]
    dX2 = v[2]

    n = ref.ang_vel #Angular velocity
    omega = np.array([0,0,n]) #Angular velocity vector in LVLH frame

    #Calculate Chief and deputy states in ECI frame at the time t

    pos_ref,vel_ref,LVLH,Base = ref.ref_orbit_pos(t)
    pos_dep = orbits.LVLH_Sat(r,v,t,ref).to_ECI().pos

    """ J2 Acceleration """
    J2_p = J2_pert(pos_dep,pos_ref,ref.R_orb) #Calculate J2 in ECI frame
    LVLH_J2_p = np.dot(LVLH,J2_p) #Convert to LVLH frame

    #HCW Equations (second order correction, see Butcher 16)
    K = np.diag(np.array([3*n**2,0,-(n**2)]))
    Gamma2 = n**2/ref.R_orb*np.array([-3*r[0]**2 + 1.5*r[1]**2 + 1.5*r[2]**2, 3*r[0]*r[1], 3*r[0]*r[2]])
    Gamma3 = (n/ref.R_orb)**2*np.array([4*r[0]**3-6*r[0]*(r[1]**2+r[2]**2),-6*r[0]**2*r[1]+1.5*r[1]**3+1.5*r[1]*r[2]**2,-6*r[0]**2*r[2]+1.5*r[2]**3+1.5*r[2]*r[1]**2])
    Gamma2 = 0
    Gamma3 = 0

    #Position vector of deputy
    #rd = np.array([ECI.R_orb+r[0],r[1],r[2]])
    #Acceleration vector - analytical version (See Butcher 18)
    #a = -2*np.cross(omega,v) - np.cross(omega,np.cross(omega,rd)) - const.GM_earth.value*rd/np.linalg.norm(rd)**3  + LVLH_J2_p + LVLH_solar_p + LVLH_drag_p
    #LVLH_J2_p = 0

    #Acceleration is the HCW Equations, plus the required perturbations
    a = -2*np.cross(omega,v) + np.matmul(K,r) + Gamma2 + LVLH_J2_p + Gamma3

    #Print kinetic energy while integrating
    #print(r[0] + v[0]/n)

    #Second half of the differential vector (derivative of velocity, acceleration)
    dX3 = a[0]
    dX4 = a[1]
    dX5 = a[2]
    return np.array([dX0,dX1,dX2,dX3,dX4,dX5])
