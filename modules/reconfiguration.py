# DelV Recongfiguration function #

import numpy as np
import astropy.constants as const
import modules.quaternions as qt
from modules.orbits import init_deputy

""" Calculate Delta V requirement to reconfigure an orbit from ECI1 to ECI2 """
def del_v_reconfigure(ECI1,ECI2):
    h0 = ECI1.h_0 #Angular momentum vector

    #Crossing point vectors on a unit sphere
    point1 = np.cross(qt.rotate(h0,ECI1.q1),qt.rotate(h0,ECI2.q1))
    point2 = np.cross(qt.rotate(h0,ECI1.q2),qt.rotate(h0,ECI2.q2))
    
    if not np.any(point1):
        point1 = init_deputy(ECI1,1,time=0).pos
    if not np.any(point2):
        point2 = init_deputy(ECI1,2,time=0).pos
    
    point1 /= (np.linalg.norm(point1)/ECI1.R_orb)
    point2 /= (np.linalg.norm(point2)/ECI1.R_orb)

    ang_vel = ECI1.ang_vel
    period = ECI1.period


    #Given a deputy position, calculate the time it was at that point
    def deputy_pos_to_time(ECI,pos,q):
        chief_pos = qt.rotate(pos,qt.conjugate(q)) #Position of the chief
        flat_chief = qt.rotate(chief_pos,qt.conjugate(ECI.q0)) #Position of the chief on the xy plane orbit
        angle = np.arctan2(flat_chief[1],flat_chief[0]) #Phase angle of chief position
        return angle/ang_vel

    t_a1 = deputy_pos_to_time(ECI1,point1,ECI1.q1)
    t_a2 = deputy_pos_to_time(ECI1,point2,ECI1.q2)
    t_b1 = deputy_pos_to_time(ECI2,point1,ECI2.q1)
    t_b2 = deputy_pos_to_time(ECI2,point2,ECI2.q2)

    if t_b1 - t_a1 > 0.5*period:
        t_a1 += period
    elif t_b1 - t_a1 < -0.5*period:
        t_b1 += period

    if t_b2 - t_a2 > 0.5*period:
        t_a2 += period
    elif t_b2 - t_a2 < -0.5*period:
        t_b2 += period

    mu = const.GM_earth.value

    #Vis viva equation
    def vis_viva(r,a):
        return np.sqrt(mu*(2/r - 1/a))

    #Calculate delta v from phase difference
    del_t1 = t_b1 - t_a1
    #import pdb; pdb.set_trace()
    T1 = del_t1 + period
    a1 = (mu*(T1/(2*np.pi))**2)**(1/3)
    del_v1 = 2*np.abs(vis_viva(ECI1.R_orb,a1) - vis_viva(ECI1.R_orb,ECI1.R_orb))

    del_t2 = t_b2 - t_a2
    T2 = del_t2 + period
    a2 = (mu*(T2/(2*np.pi))**2)**(1/3)
    del_v2 = 2*np.abs(vis_viva(ECI1.R_orb,a2) - vis_viva(ECI1.R_orb,ECI1.R_orb))
    
    print("Phase change: " + str(del_v1) + " , " + str(del_v2))

    #Caclulate delta_v from inclination change
    vel_a1 = init_deputy(ECI1,1,time=t_a1).vel
    vel_a2 = init_deputy(ECI1,2,time=t_a2).vel
    vel_b1 = init_deputy(ECI2,1,time=t_b1).vel
    vel_b2 = init_deputy(ECI2,2,time=t_b2).vel

    print("Inc change: " + str(np.linalg.norm(vel_b1-vel_a1)) + " , " + str(np.linalg.norm(vel_b2-vel_a2)))

    del_v1 += np.linalg.norm(vel_b1-vel_a1)
    del_v2 += np.linalg.norm(vel_b2-vel_a2)
    
    print("Total change: " + str(del_v1) + " , " + str(del_v2))

    return del_v1, del_v2
