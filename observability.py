### Observability Helper functions ###

import numpy as np
import matplotlib.path as mpltPath
import astropy.constants as const

def find_anti_sun_vector(t):
    oblq = np.radians(23.4) #Obliquity of the ecliptic
    Earth_sun_ang_vel = 2*np.pi/(365.25*24*60*60) #Angular velocity of the Earth around the Sun
    phase = t*Earth_sun_ang_vel
    pos = np.array([np.cos(phase)*np.cos(oblq),np.sin(phase),np.cos(phase)*np.sin(oblq)])
    return pos
        
# s is star vector, angle is angle within antisun
def check_sun(s,t,angle):
    anti_sun = find_anti_sun_vector(t)
    return np.arccos(np.dot(anti_sun,s)) < angle

def point_in_polygon_circle(p,r,n_points):
    phases = np.linspace(0,2*np.pi,n_points)
    circ_pts = np.array([r*np.cos(phases),r*np.sin(phases)]).transpose()
    
    path = mpltPath.Path(circ_pts)
    return path.contains_point(p)


def check_earth(dep1,dep2,s):
    r_E = const.R_earth.value
    def check_deputy(dep_pos):
        proj = dep_pos + s*np.abs(np.dot(dep_pos,s))
        if np.dot(proj,s) != 0:
            return True
        else:
            if (s == np.array([0,0,1])).all():
                uhat = np.array([0,1,0])
            else:
                uhat = np.cross(s,np.array([0,0,1]))
            vhat = np.cross(s,uhat)
        
            rotmat = np.array([uhat,vhat,s])
            proj_rot = np.dot(rotmat,proj)
            proj_2d = proj_rot[:2]
            return not point_in_polygon_circle(proj_2d,r_E,1000)
    
    check_dep1 = check_deputy(dep1[:3])
    check_dep2 = check_deputy(dep2[:3])
    return (check_dep1 and check_dep2)

def check_obs(t,dep1,dep2,s,angle):
    return check_sun(s,t,angle) and check_earth(dep1,dep2,s)