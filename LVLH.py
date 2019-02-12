import numpy as np
import quaternions as qt

def to_LVLH(v_c,q):
  x = np.array([1,0,0])
  y = np.array([0,1,0])
  z = np.array([0,0,1])

  X = qt.rotate(x,q)
  Y = qt.rotate(y,q)
  Z = qt.rotate(z,q)

  r = np.linalg.norm(v_c)

  rot_mat = np.array([X,Y,Z])
  A1 = np.array([[v_c[0]/r,v_c[1]/r,0],[v_c[1]/r,-v_c[0]/r,0],[0,0,1]])

  A2 = np.matmul(A1,rot_mat)

  def LVLH(v):
      return np.dot(A2,v)-np.dot(A2,v_c)

  return LVLH

def orbits_to_LVLH(chief,deputy1,deputy2,star,chiefq):
    func = to_LVLH(chief,chiefq)
    new_chief = func(chief)
    new_deputy1 = func(deputy1)
    new_deputy2 = func(deputy2)
    new_star = func(star)
    return new_chief, new_deputy1, new_deputy2,new_star
