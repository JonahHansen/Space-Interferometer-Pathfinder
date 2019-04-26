# DelV Recongfiguration function #

import numpy as np
import astropy.constants as const
import quaternions as qt

""" Calculate Delta V requirement to reconfigure an orbit from ECI1 to ECI2 """
def del_v_reconfigure(ECI1,ECI2,n_times):
    times = np.linspace(0,ECI2.period,n_times)
    h0 = ECI1.h_0 #Angular momentum vector
    
    #Crossing point vectors on a unit sphere
    point1 = np.cross(qt.rotate(h0,ECI1.q1),qt.rotate(h0,ECI2.q1))
    point2 = np.cross(qt.rotate(h0,ECI1.q2),qt.rotate(h0,ECI2.q2))
    point1 /= (np.linalg.norm(point1)/ECI1.R_orb)
    point2 /= (np.linalg.norm(point2)/ECI1.R_orb)

    pos_11_ls = []
    pos_12_ls = []
    pos_21_ls = []
    pos_22_ls = []

    #Find the times at which each orbit for the two deputies is closest
    for t in times:
        c = ECI1.chief_state(t)
        d11 = ECI1.deputy1_state(c)[:3]
        d12 = ECI1.deputy2_state(c)[:3]
        d21 = ECI2.deputy1_state(c)[:3]
        d22 = ECI2.deputy2_state(c)[:3]

        pos_11_ls.append(np.linalg.norm(d11 - point1))
        pos_12_ls.append(np.linalg.norm(d12 - point2))
        pos_21_ls.append(np.linalg.norm(d21 - point1))
        pos_22_ls.append(np.linalg.norm(d22 - point2))

    t_11 = times[np.array(pos_11_ls).argmin()]
    t_12 = times[np.array(pos_12_ls).argmin()]
    t_21 = times[np.array(pos_21_ls).argmin()]
    t_22 = times[np.array(pos_22_ls).argmin()]

    print(t_11,t_12,t_21,t_22)

    mu = const.GM_earth.value

    #Vis viva equation
    def vis_viva(r,a):
        return np.sqrt(mu*(2/r - 1/a))

    #Calculate delta v from phase difference
    del_t1 = t_21 - t_11
    T1 = del_t1 + ECI1.period
    a1 = (mu*(T1/(2*np.pi))**2)**(1/3)
    del_v1 = np.abs(vis_viva(ECI1.R_orb,a1) - vis_viva(ECI1.R_orb,ECI1.R_orb))

    del_t2 = t_22 - t_12
    T2 = del_t2 + ECI1.period
    a2 = (mu*(T2/(2*np.pi))**2)**(1/3)
    del_v2 = np.abs(vis_viva(ECI1.R_orb,a2) - vis_viva(ECI1.R_orb,ECI1.R_orb))

    print(del_v1,del_v2)

    #Caclulate delta_v from inclination change
    vel_11 = ECI1.deputy1_state(ECI1.chief_state(t_11))[3:]
    vel_12 = ECI1.deputy1_state(ECI1.chief_state(t_12))[3:]
    vel_21 = ECI2.deputy1_state(ECI2.chief_state(t_21))[3:]
    vel_22 = ECI2.deputy1_state(ECI2.chief_state(t_22))[3:]

    del_v1 += np.linalg.norm(vel_21-vel_11)
    del_v2 += np.linalg.norm(vel_22-vel_12)
    print(del_v1,del_v2)
    return del_v1, del_v2
