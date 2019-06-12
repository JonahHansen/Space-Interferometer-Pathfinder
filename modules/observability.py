### Observability Helper functions ###

import numpy as np
import matplotlib.path as mpltPath
import astropy.constants as const

""" Find the anti-sun vector at a given time """
def find_anti_sun_vector(t):
    oblq = np.radians(23.4) #Obliquity of the ecliptic
    Earth_sun_ang_vel = 2*np.pi/(365.25*24*60*60) #Angular velocity of the Earth around the Sun
    phase = t*Earth_sun_ang_vel
    pos = np.array([np.cos(phase)*np.cos(oblq),np.sin(phase),np.cos(phase)*np.sin(oblq)])
    return pos

""" Check if star vector is within antisun """
# s is star vector, angle is angle within antisun
def check_sun(s,t,angle):
    anti_sun = find_anti_sun_vector(t)
    return np.arccos(np.dot(anti_sun,s)) < angle

""" Check if point p is within a circle of radius r centred at 0 """
def point_in_polygon_circle(p,r,n_points):
    phases = np.linspace(0,2*np.pi,n_points)

    #Create polygon
    circ_pts = np.array([r*np.cos(phases),r*np.sin(phases)]).transpose()
    path = mpltPath.Path(circ_pts)
    return path.contains_point(p)

""" Check if Earth is blocking field of view """
def check_earth(dep1,dep2,R_orb,s,UVmat):
    r_E = const.R_earth.value
    def check_deputy(dep_pos):
        #Project position onto a plane perpendicular to star vector
        #centred at the centre of the Earth
        #proj = dep_pos + s*np.abs(np.dot(dep_pos,s))

        #Check if position is in front of Earth
        if np.dot(dep_pos,s) > 0:
            return True

        else:
            #Rotate axes to UV frame
            proj_rot = np.dot(UVmat,dep_pos)
            proj_2d = proj_rot[:2]

            new_rad = np.sqrt(R_orb**2 - r_E**2)*np.tan(np.arccos(r_E/R_orb)/3) + r_E
            #Check if position lies within Earth
            return not point_in_polygon_circle(proj_2d,new_rad,1000)

    check_dep1 = check_deputy(dep1.pos)
    check_dep2 = check_deputy(dep2.pos)
    return (check_dep1 and check_dep2)

""" Combine both checks """
def check_obs(t,dep1,dep2,antisun_angle,ECI):
    return check_sun(ECI.s_hat,t,antisun_angle) and check_earth(dep1,dep2,ECI.R_orb,ECI.s_hat,ECI.UVmat)
