""" Frames module"""

import numpy as np
import quaternions as qt

#Creates a function that transforms from geocentric cartesian coordinates (ECEF) to LVLH frame
# Takes r_c (position of chief satellite) and q (rotation of chief orbital plane)
def to_LVLH_func(r_c,q):

  h_hat = qt.rotate(np.array([0,0,1]),q) #Angular momentum vector (rotated "z" axis)
  r_hat = r_c/np.linalg.norm(r_c) #Position vector pointing away from the centre of the Earth
  v_hat = np.cross(h_hat,r_hat) #Velocity vector pointing counter-clockwise

  rot_mat = np.array([r_hat,v_hat,h_hat]) #Rotation matrix from three unit vectors

  #Function to take a vector in ECEF and return it in LVLH
  def LVLH(v):
      return np.dot(rot_mat,v)-np.dot(rot_mat,r_c)

  return LVLH

#Convert chief,deputies and star orbits into LVLH frame
def orbits_to_LVLH(r_c,other_vectors,chiefq):
    func = to_LVLH_func(r_c,chiefq) #Create change of basis function

    #New vectors in LVLH frame
    return_ls = [func(r_c)]
    for vec in other_vectors:
        return_ls.append(func(vec))

    return np.array(return_ls)

#Creates a function that transforms from geocentric cartesian coordinates (ECEF) to Baseline frame
# Takes r_c (position of chief satellite), r_d2 (position of second/forward deputy) and star direction s_hat
def to_baseline_func(r_c,r_d2,s_hat):
    b_hat = (r_d2 - r_c)/np.linalg.norm(r_d2 - r_c) #Direction along baseline
    k_hat = np.cross(s_hat,b_hat) #Other direction

    rot_mat = np.array([k_hat,b_hat,s_hat]) #Create rotation matrix

    #Function to take a vector in ECEF and return it in Baseline
    def baseline(v):
        return np.dot(rot_mat,v)-np.dot(rot_mat,r_c)

    return baseline

#Convert chief, deputies, star and other vectors into Baseline frame
def orbits_to_baseline(r_c,r_d1,r_d2,s_hat,other_vectors):
    func = to_baseline_func(r_c,r_d2,s_hat) #Change of basis function

    #New vectors in Baseline frame
    return_ls = [func(r_c),func(r_d1),func(r_d2)]
    for vec in other_vectors:
        return_ls.append(func(vec))

    return np.array(return_ls)
