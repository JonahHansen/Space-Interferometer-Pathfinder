""" Quaternion Module """
"""
TO USE:
1. First define rotation using to_q
2. Add more rotations by multiplying quaternions
3. Use either rotate or rotate_points to apply final quaternion rotation
"""
import numpy as np

class quaternion:
    #quaternion class. s is the scalar part, v is the vector part.
    def __init__(self,scalar,vector):
        self.s = scalar
        self.v = vector

    #Normalise the quaternion
    def normalise(self):
        norm = np.sqrt(self.s**2 + np.dot(self.v,self.v))
        self.s = self.s/norm
        self.v = self.v/norm
        return

#Conjugate of a quaternion
def conjugate(q):
    q2 = quaternion(q.s,q.v)
    q2.v = -(q2.v)
    return q2

#Take two normalised quaternions (rotations), and multiply them
def q_mult(q1,q2):
    q3 = quaternion(0,np.array([0,0,0]))
    q3.s = q1.s*q2.s - np.dot(q1.v,q2.v)
    q3.v = q1.s*q2.v + q2.s*q1.v + np.cross(q1.v,q2.v)
    return q3

#Turn a rotation into a normalised quaternion (angle in radians). Axis does not need to be normalised
def to_q(axis,angle):
    axis_norm = np.linalg.norm(axis)
    angle = angle/2.
    q = quaternion(np.cos(angle),axis/axis_norm*np.sin(angle))
    q.normalise()
    return q

#Rotate a vector by the quaternion
def rotate(v,q):
    p = quaternion(0,v)
    result_q = q_mult(q_mult(q,p),conjugate(q))
    return result_q.v

#Get axis and angle from quaternion
def from_q(q):
    norm_v = np.linalg.norm(q.v)
    theta = 2*np.arctan2(norm_v,q.s)
    if theta == 0:
        v = [0,0,0]
    else:
        v = q.v/norm_v
    return v, theta

#Rotate all points in a list of positions
def rotate_points(ls,q):
    new_ls = np.zeros(ls.shape)
    for i in range(len(ls)):
        new_ls[i] = rotate(ls[i],q)
    return new_ls

#Combine two rotations (quaternions)
def comb_rot(first,second):
    return q_mult(second,first)
