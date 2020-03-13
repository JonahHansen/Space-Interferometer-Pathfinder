### Observability Helper functions ###

import numpy as np
import matplotlib.path as mpltPath
import astropy.constants as const

"""
Find the antisolar vector at a given time
Inputs:
    t - time to find antisolar vector
Outputs:
    Antisolar vector
"""
def find_anti_sun_vector(t):
    oblq = np.radians(23.4) #Obliquity of the ecliptic
    Earth_sun_ang_vel = 2*np.pi/(365.25*24*60*60) #Angular velocity of the Earth around the Sun
    phase = t*Earth_sun_ang_vel
    pos = np.array([np.cos(phase),np.sin(phase)*np.cos(oblq),np.sin(phase)*np.sin(oblq)])
    return pos


"""
Check if star vector is within antisun
Inputs:
    s - star vector
    t - time
    angle - antisolar angle
Output:
    Boolean as to whether star is observable
"""
def check_sun(s,t,angle):
    #Find antisolar vector
    anti_sun = find_anti_sun_vector(t)
    return np.arccos(np.dot(s,anti_sun)) < angle


"""
Check if Earth is blocking field of view
Inputs:
    pos - position of the satellite
    R_orb - radius of the orbit
    s - star vector
    multi - are we inputing multiple star vectors to check?
Outputs:
    Boolean as to whether star is observable
"""
def check_earth(pos,R_orb,s,multi):
    #Earth radius
    r_E = const.R_earth.value
    #Extend Earth radius
    new_rad = np.sqrt(R_orb**2 - r_E**2)*np.tan(np.arccos(r_E/R_orb)/3) + r_E
    #Perform the check
    dot = np.dot(s,pos)
    if multi:
        p = np.linalg.norm(pos - dot[:,:,np.newaxis]*s,axis=2)
    else:
        p = np.linalg.norm(pos - dot*s)
    return np.logical_or(dot > 0, p > new_rad)


"""
Combine both checks
Inputs:
    t - time
    s - star vector
    pos - position of satellite
    antisolar angle - as stated
    ref - reference orbit
    multi - are we inputing multiple star vectors
Outputs:
    Boolean as to whether star is observable
"""
def check_obs(t,s,pos,antisolar_angle,ref,multi=True):
    return np.logical_and(check_sun(s,t,antisolar_angle),check_earth(pos,ref.R_orb,s,multi))
