import numpy as np
import astropy.constants as const
from scipy.optimize import fsolve
import modules.quaternions as qt


""" J2 Perturbation function from Schweighart's paper """
def J2_rel_pet(sat,ref,n):

    #DEFINE VARIABLES AS IN PAPER
    r_ref = ref.R_orb
    i_ref = ref.inc_0

    if n == 1:
        q = ref.q1
    elif n == 2:
        q = ref.q2

    h_dep = qt.rotate(ref.h_0,q)
    i_dep = np.arccos(h_dep[2]/np.linalg.norm(h_dep))

    J2 = 0.00108263
    R_e = const.R_earth.value

    [x_0,y_0,z_0] = sat.pos
    dz_0 = sat.vel[2]

    c = ref.Sch_c
    n = ref.ang_vel
    k = ref.Sch_k
    
    i_sat1 = dz_0/(k*r_ref)+i_dep

    if i_ref == 0:
        omega_0 = 0
    else:
        omega_0 = z_0/(r_ref*np.sin(i_ref))

    if (omega_0 and i_dep) != 0:
        gamma_0 = np.pi/2 - np.arctan((1/np.tan(i_dep)*np.sin(i_sat1)-np.cos(i_sat1)*np.cos(omega_0))/np.sin(omega_0))
    else:
        gamma_0 = 0
    phi_0 = np.arccos(np.cos(i_sat1)*np.cos(i_dep)+np.sin(i_sat1)*np.sin(i_dep)*np.cos(omega_0))
    d_omega_sat1 = -3*n*J2*R_e**2/(2*r_ref**2)*np.cos(i_sat1)
    d_omega_dep = -3*n*J2*R_e**2/(2*r_ref**2)*np.cos(i_dep)

    temp = np.cos(gamma_0)*np.sin(gamma_0)*1/np.tan(omega_0)
    temp = temp if temp == temp else 0

    q = n*c - (temp-np.sin(gamma_0)**2*np.cos(i_sat1))*(d_omega_sat1 - d_omega_dep)-d_omega_sat1*np.cos(i_sat1)
    l = -r_ref*np.sin(i_sat1)*np.sin(i_dep)*np.sin(omega_0)/np.sin(phi_0)*(d_omega_sat1-d_omega_dep)
    l = l if l == l else 0

    def equations(p):
        m,phi = p
        return(m*np.sin(phi)-z_0,l*np.sin(phi)+q*m*np.cos(phi)-dz_0)

    #Solve simultaneous equations
    m,phi = fsolve(equations,(1,1))

    #Equations of motion
    def J2_pet_func(t,state):
        [x,y,z] = state[:3] #Position
        [dx,dy,dz] = state[3:] #Velocity
        dX0 = dx
        dX1 = dy
        dX2 = dz

        #Gamma2 = n**2/r_ref*np.array([-3*x**2 + 1.5*y**2 + 1.5*z**2, 3*x*y, 3*x*z])
        #Gamma3 = (n/r_ref)**2*np.array([4*x**3-6*x*(y**2+z**2),-6*x**2*y+1.5*y**3+1.5*y*z**2,-6*x**2*z+1.5*z**3+1.5*z*y**2])
        Gamma2 = np.array([0,0,0])
        Gamma3 = np.array([0,0,0])

        dX3 = 2*n*c*dy + (5*c**2-2)*n**2*x + Gamma2[0] + Gamma3[0]
        dX4 = -2*n*c*dx + Gamma2[1] + Gamma3[1]
        dX5 = -q**2*z + 2*l*q*np.cos(q*t+phi) + Gamma2[2] + Gamma3[2]
        #print(x + dx/n) #Energy

        return np.array([dX0,dX1,dX2,dX3,dX4,dX5])

    return J2_pet_func

""" J2 Perturbation function from Schweighart's paper """
def J2_pet(sat,ref):

    #DEFINE VARIABLES AS IN PAPER
    r_ref = ref.R_orb
    i_ref = ref.inc_0


    J2 = 0.00108263
    R_e = const.R_earth.value

    [x_0,y_0,z_0] = sat.pos
    dz_0 = sat.vel[2]

    c = ref.Sch_c
    n = ref.ang_vel
    k = ref.Sch_k
    
    i_sat = dz_0/(k*r_ref)+i_ref

    if i_ref == 0:
        omega_0 = 0
    else:
        omega_0 = z_0/(r_ref*np.sin(i_ref))

    if (omega_0 and i_ref) != 0:
        gamma_0 = np.pi/2 - np.arctan((1/np.tan(i_ref)*np.sin(i_sat)-np.cos(i_sat)*np.cos(omega_0))/np.sin(omega_0))
    else:
        gamma_0 = 0
        
    phi_0 = np.arccos(np.cos(i_sat)*np.cos(i_ref)+np.sin(i_sat)*np.sin(i_ref)*np.cos(omega_0))
    d_omega_sat = -3*n*J2*R_e**2/(2*r_ref**2)*np.cos(i_sat)
    d_omega_ref = -3*n*J2*R_e**2/(2*r_ref**2)*np.cos(i_ref)

    temp = np.cos(gamma_0)*np.sin(gamma_0)*1/np.tan(omega_0)
    temp = temp if temp == temp else 0

    q = n*c - (temp-np.sin(gamma_0)**2*np.cos(i_sat))*(d_omega_sat - d_omega_ref)-d_omega_sat*np.cos(i_sat)
    l = -r_ref*np.sin(i_sat)*np.sin(i_ref)*np.sin(omega_0)/np.sin(phi_0)*(d_omega_sat-d_omega_ref)
    l = l if l == l else 0

    def equations(p):
        m,phi = p
        return(m*np.sin(phi)-z_0,l*np.sin(phi)+q*m*np.cos(phi)-dz_0)

    #Solve simultaneous equations
    m,phi = fsolve(equations,(1,1))

    #Equations of motion
    def J2_pet_func(t,state):
        [x,y,z] = state[:3] #Position
        [dx,dy,dz] = state[3:] #Velocity
        dX0 = dx
        dX1 = dy
        dX2 = dz

        #Gamma2 = n**2/r_ref*np.array([-3*x**2 + 1.5*y**2 + 1.5*z**2, 3*x*y, 3*x*z])
        #Gamma3 = (n/r_ref)**2*np.array([4*x**3-6*x*(y**2+z**2),-6*x**2*y+1.5*y**3+1.5*y*z**2,-6*x**2*z+1.5*z**3+1.5*z*y**2])
        Gamma2 = np.array([0,0,0])
        Gamma3 = np.array([0,0,0])

        dX3 = 2*n*c*dy + (5*c**2-2)*n**2*x - 3*n**2*J2*(R_e**2/r_ref)*(0.5 - (3*np.sin(i_ref)**2*np.sin(k*t)**2/2) - ((1+3*np.cos(2*i_ref))/8)) + Gamma2[0] + Gamma3[0]
        dX4 = -2*n*c*dx - 3*n**2*J2*(R_e**2/r_ref)*np.sin(i_ref)**2*np.sin(k*t)*np.cos(k*t) + Gamma2[1] + Gamma3[1]
        dX5 = -q**2*z + 2*l*q*np.cos(q*t+phi) + Gamma2[2] + Gamma3[2]
        #print(x + dx/n) #Energy

        return np.array([dX0,dX1,dX2,dX3,dX4,dX5])

    return J2_pet_func
