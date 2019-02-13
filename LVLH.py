""" LVLH module""" 

import numpy as np
import quaternions as qt

#Creates a function that transforms from geocentric cartesian coordinates (ECEF) to LVLH frame
# Takes r_c (position of chief satellite) and q (rotation of chief orbital plane)
def to_LVLH_func(r_c,q):

  h_hat = qt.rotate(np.array([0,0,1]),q) #Angular momentum vector (rotated "z" axis)
  r_hat = r_c/np.linalg.norm(r_c) #Position vector pointing away from the centre of the Earth
  v_hat = np.cross(h_hat,r_hat) #Velocity vector pointing counter-clockwise

  rot_mat = np.linalg.inv(np.transpose([r_hat,v_hat,h_hat])) #Rotation matrix from three unit vectors

  #Function to take a vector in ECEF and return it in LVLH
  def LVLH(v):
      return np.dot(rot_mat,v)-np.dot(rot_mat,r_c)

  return LVLH

#Convert chief,deputies and star orbits into LVLH frame
def orbits_to_LVLH(chief,other_vectors,chiefq):
    func = to_LVLH_func(chief,chiefq) #Create change of basis function

    #New vectors in lvlh frame
    return_ls = [func(chief)]
    for vec in other_vectors:
        return_ls.append(func(vec))

    return np.array(return_ls)

"""
def to_LVLH(v_c,q):
  x = np.array([1,0,0])
  y = np.array([0,1,0])
  z = np.array([0,0,1])

  X = qt.rotate(x,q)
  Y = qt.rotate(y,q)
  Z = qt.rotate(z,q)

  r = np.linalg.norm(v_c)

  rot_mat = np.linalg.inv(np.transpose([X,Y,Z]))
  print(np.dot(rot_mat,v_c))


  A1 = np.linalg.inv([[v_c[0]/r,-v_c[1]/r,0],[v_c[1]/r,v_c[0]/r,0],[0,0,1]])

  A2 = np.matmul(A1,rot_mat)

  def LVLH(v):
      return np.dot(A2,v)-np.dot(A2,v_c)

  return LVLH
"""
