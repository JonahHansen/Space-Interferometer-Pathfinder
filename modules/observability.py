### Observability Helper functions ###

import numpy as np
import matplotlib.path as mpltPath
import astropy.constants as const

""" Find the anti-sun vector at a given time """
def find_anti_sun_vector(t):
    oblq = np.radians(23.4) #Obliquity of the ecliptic
    Earth_sun_ang_vel = 2*np.pi/(365.25*24*60*60) #Angular velocity of the Earth around the Sun
    phase = t*Earth_sun_ang_vel
    pos = np.array([np.cos(phase),np.sin(phase)*np.cos(oblq),np.sin(phase)*np.sin(oblq)])
    return pos

""" Check if star vector is within antisun """
# s is star vector, angle is angle within antisun
def check_sun(s,t,angle):
    anti_sun = find_anti_sun_vector(t)
    return np.arccos(np.dot(s,anti_sun)) < angle

""" Check if Earth is blocking field of view """
def check_earth(pos,R_orb,s,multi):
    r_E = const.R_earth.value
    new_rad = np.sqrt(R_orb**2 - r_E**2)*np.tan(np.arccos(r_E/R_orb)/3) + r_E
    dot = np.dot(s,pos)
    if multi:
        p = np.linalg.norm(pos - dot[:,:,np.newaxis]*s,axis=2)
    else:
        p = np.linalg.norm(pos - dot*s)
    return np.logical_or(dot > 0, p > new_rad)

""" Combine both checks """
def check_obs(t,s,pos,antisun_angle,ref,multi=True):
    return np.logical_and(check_sun(s,t,antisun_angle),check_earth(pos,ref.R_orb,s,multi))
