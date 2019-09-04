from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp
import modules.orbits as orbits
import sys
from matplotlib.collections import LineCollection

plt.ion()

alt = 500e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(90) #20
#Longitude of the Ascending Node
Om_0 = np.radians(90) #0

#Stellar vector
ra = np.radians(0) #90
dec = np.radians(45)#-40

#The max distance to the other satellites in m
delta_r_max = 0.3e3

#Tolerance and steps required for the integrator
rtol = 1e-12
atol = 1e-18

zeta = float(sys.argv[2])

def dX_dt(t,state,ref):
    [x,y,z] = state[:3] #Position
    v = state[3:] #Velocity

    #First half of the differential vector (derivative of position, velocity)
    dX0 = v[0]
    dX1 = v[1]
    dX2 = v[2]

    J2 = 0.00108263 #J2 Parameter

    r = np.linalg.norm(state[:3])

    #Calculate J2 acceleration from the equation in ECI frame
    J2_fac1 = 3/2*J2*const.GM_earth.value*const.R_earth.value**2/r**5
    J2_fac2 = 5*z**2/r**2
    J2_p = J2_fac1*np.array([x*(J2_fac2-1),y*(J2_fac2-1),z*(J2_fac2-3)])

    r_hat = state[:3]/r
    a = -const.GM_earth.value/r**2*r_hat + J2_p
    dX3 = a[0]
    dX4 = a[1]
    dX5 = a[2]

    return np.array([dX0,dX1,dX2,dX3,dX4,dX5])


def del_v_func(c,d1,d2,t,pt,ref):
    """
    c_s_hat = np.dot(c.vel,ref.s_hat)
    d1_s_hat = np.dot(d1.vel,ref.s_hat)
    d2_s_hat = np.dot(d2.vel,ref.s_hat)

    d1_rel_s = d1_s_hat - c_s_hat
    d2_rel_s = d2_s_hat - c_s_hat
    print(d1_rel_s)

    """
    sat0 = c.to_Baseline(state=c.state)
    sat1 = d1.to_Baseline(state=c.state)
    sat2 = d2.to_Baseline(state=c.state)

    csat = sat0.state
    dsat1 = sat1.state
    dsat2 = sat2.state

    delvs1 = np.zeros(3)
    delvs2 = np.zeros(3)
    delvs0 = np.zeros(3)

    """
    #delvs1[2] = -dsat1[2]/(t-pt)
    #delvs2[2] = -dsat2[2]/(t-pt)
    #import pdb; pdb.set_trace()
    #delvs1 = -1*d1_rel_s/(t-pt)
    #delvs2 = -1*d2_rel_s/(t-pt)

    delvs1_sc = -dsat1[2]/(t-pt)
    delvs2_sc = -dsat2[2]/(t-pt)

    if delvs1_sc < 0:
        if delvs2_sc < 0:
            delvs1[2] = -delvs2_sc
            delvs0[2] = -(delvs1_sc + delvs2_sc)
            delvs2[2] = -delvs1_sc
        else:
            delvs2[2] -= delvs1_sc
            delvs0[2] -= delvs1_sc
    elif delvs2_sc < 0:
            delvs1[2] -= delvs2_sc
            delvs0[2] -= delvs2_sc

    """

    max_s_sep = np.max([csat[2],dsat1[2],dsat2[2]])
    #print(max_s_sep)

    del_t = (t-pt)/zeta

    delvs0[2] = (max_s_sep - csat[2])/del_t
    delvs1[2] = (max_s_sep - dsat1[2])/del_t
    delvs2[2] = (max_s_sep - dsat2[2])/del_t

    b1 = np.linalg.norm(dsat1[0:2])
    b2 = np.linalg.norm(dsat2[0:2])
    del_b = b2-b1
    #print(del_b)

    if del_b >= 0:
        delvs2[2] += del_b/del_t
    elif del_b < 0:
        delvs1[2] += -del_b/del_t

    """
    #Calculate position vector to the midpoint of the two deputies
    del_b = dsat1[0:3] - dsat2[0:3] #Separation vector
    del_b_half = 0.5*del_b #Midpoint
    m0 = dsat1[0:3] + del_b_half #Midpoint from centre
    m0[2] = 00

    delvb = m0/(t-pt)
    delvb = np.array([0,0,0])
    """

    delv = np.array([delvs0[2],delvs1[2],delvs2[2]])
    #print(delv)
    sat0.vel += delvs0
    sat1.vel += delvs1
    sat2.vel += delvs2

    new_sat0 = sat0.to_ECI(state = c.state).state
    new_sat1 = sat1.to_ECI(state = c.state).state
    new_sat2 = sat2.to_ECI(state = c.state).state
    """
    #import pdb; pdb.set_trace()
    c.state[3:] += delvs0*ref.s_hat
    d1.state[3:] += delvs1*ref.s_hat
    d2.state[3:] += delvs2*ref.s_hat

    new_sat0 = c.state
    new_sat1 = d1.state
    new_sat2 = d2.state
    """

    return delv,new_sat0,new_sat1,new_sat2


#------------------------------------------------------------------------------------------
#Calculate reference orbit, in the geocentric (ECI) frame (See Orbit module)
ref = orbits.Reference_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

#Number of orbits
n_orbits = 0.5
#Number of phases in each orbit
n_phases = 500
#Total evaluation points
n_times = int(n_orbits*n_phases)
times = np.linspace(0,ref.period*n_orbits,n_times) #Create list of times

#Initial states of the satellites
chief_0 = orbits.init_chief(ref).state
deputy1_0 = orbits.init_deputy(ref,1).state
deputy2_0 = orbits.init_deputy(ref,2).state

chief_states = np.array([chief_0])
deputy1_states = np.array([deputy1_0])
deputy2_states = np.array([deputy2_0])
delv_bank = []

t_burn = int(sys.argv[1])
t0 = 0.0
t_bank = np.array([0])

while t0 < times[-1]:
    burn_pt = t0 + t_burn
    ts = np.linspace(t0,burn_pt,10) #Every 0.1s
    t_bank = np.append(t_bank,ts)

    #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
    X_c = solve_ivp(lambda t, y: dX_dt(t,y,ref), [ts[0],ts[-1]], chief_0, t_eval = ts, rtol = rtol, atol = atol)
    #Check if successful integration
    if not X_c.success:
        raise Exception("Integration Chief failed!!!!")

    #Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
    X_d1 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [ts[0],ts[-1]], deputy1_0, t_eval = ts, rtol = rtol, atol = atol)
    #Check if successful integration
    if not X_d1.success:
        raise Exception("Integration Deputy 1 failed!!!!")

    X_d2 = solve_ivp(lambda t, y: dX_dt(t,y,ref), [ts[0],ts[-1]], deputy2_0, t_eval = ts, rtol = rtol, atol = atol)
    if not X_d2.success:
        raise Exception("Integration Deputy 2 failed!!!!")

    chief_p_states = X_c.y.transpose()
    deputy1_p_states = X_d1.y.transpose()
    deputy2_p_states = X_d2.y.transpose()

    chief_states = np.append(chief_states,chief_p_states,axis=0)
    deputy1_states = np.append(deputy1_states,deputy1_p_states,axis=0)
    deputy2_states = np.append(deputy2_states,deputy2_p_states,axis=0)

    c = orbits.ECI_Sat(chief_p_states[-1,:3],chief_p_states[-1,3:],ts[-1],ref)
    d1 = orbits.ECI_Sat(deputy1_p_states[-1,:3],deputy1_p_states[-1,3:],ts[-1],ref)
    d2 = orbits.ECI_Sat(deputy2_p_states[-1,:3],deputy2_p_states[-1,3:],ts[-1],ref)

    delv,new_c,new_d1,new_d2 = del_v_func(c,d1,d2,burn_pt,t0,ref)
    #import pdb; pdb.set_trace()
    delv_bank.append(delv)
    chief_0 = new_c
    deputy1_0 = new_d1
    deputy2_0 = new_d2
    t0 = burn_pt
    #print(t0)

n_times = len(chief_states)

rel_p_dep1 = []
rel_p_dep2 = []

print("Integration Done")
for i in range(len(t_bank)):
    rel_p_dep1.append(orbits.ECI_Sat(deputy1_states[i,:3],deputy1_states[i,3:],t_bank[i],ref).to_Baseline(state=chief_states[i]))
    rel_p_dep2.append(orbits.ECI_Sat(deputy2_states[i,:3],deputy2_states[i,3:],t_bank[i],ref).to_Baseline(state=chief_states[i]))
print("Classifying Done")

#--------------------------------------------------------------------------------------------- #
#Separations and accelerations
baseline_sep = np.zeros(n_times) #Separation along the baseline
s_hat_drd1 = np.zeros(n_times) #Deputy1 position in star direction
s_hat_drd2 = np.zeros(n_times) #Deputy2 position in star direction
b_hat_drd1 = np.zeros(n_times) #Deputy1 position in baseline direction
b_hat_drd2 = np.zeros(n_times) #Deputy2 position in baseline direction
s_hat_sep = np.zeros(n_times) #Separation along the baseline
total_sep = np.zeros(n_times) #Total separation

for ix in range(n_times):
    #Baseline separations is simply the difference between the positions of the two deputies
    baseline_sep[ix] = np.linalg.norm(rel_p_dep2[ix].pos) - np.linalg.norm(rel_p_dep1[ix].pos)

    #Component of perturbed orbit in star direction
    s_hat_drd1[ix] = rel_p_dep1[ix].pos[2]
    s_hat_drd2[ix] = rel_p_dep2[ix].pos[2]

    #Component of perturbed orbit in baseline direction
    b_hat_drd1[ix] = rel_p_dep1[ix].pos[0]
    b_hat_drd2[ix] = rel_p_dep2[ix].pos[0]

    #Separation of the two deputies in the star direction
    s_hat_sep[ix] = s_hat_drd2[ix] - s_hat_drd1[ix]
    #baseline_sep[ix] = b_hat_drd1[ix] + b_hat_drd2[ix]
    #Delta of the separation along the star direction and the baseline direction
    total_sep[ix] = baseline_sep[ix] - s_hat_sep[ix]

print(np.sum(np.array(delv_bank),axis=0))
print(np.max(np.abs(total_sep)))

# ---------------------------------------------------------------------- #
### PLOTTING STUFF ###

#Plot separation along the star direction
plt.figure(3)
plt.clf()
plt.plot(t_bank,s_hat_drd1,"b-",label="Deputy 1, s direction")
plt.plot(t_bank,s_hat_drd2,"g-",label="Deputy 2, s direction")
plt.plot(t_bank,s_hat_sep,"r-",label="Separation, s direction")
plt.xlabel("Times(s)")
plt.ylabel("Separation(m)")
plt.title('Separations against time due to perturbations')
plt.legend()

#Plot separation along the baseline direction
plt.figure(4)
plt.clf()
plt.plot(t_bank,baseline_sep,"y-",label="Separation, baseline direction")
#plt.plot(times,total_sep,"c-",label="Total direction")
plt.xlabel("Times(s)")
plt.ylabel("Separation(m)")
plt.title('Separations against time due to perturbations')
plt.legend()

#Plot separation in the baseline frame
plt.figure(5)
plt.clf()
plt.plot(t_bank,total_sep,"c-",label="Total direction")
plt.xlabel("Times(s)")
plt.ylabel("Separation(m)")
plt.title('Separations against time due to perturbations')
plt.legend()
