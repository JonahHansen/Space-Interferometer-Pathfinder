from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import astropy.constants as const
from scipy.integrate import solve_ivp, odeint
from modules.orbits import ECI_orbit, Chief, init_deputy
from matplotlib.collections import LineCollection
from modules.Schweighart_J2 import J2_pet
#from scipy.optimize import minimize

plt.ion()

alt = 500e3 #In m
R_e = const.R_earth.value  #In m
#Orbital radius the sum of earth radius and altitude
R_orb = R_e + alt

#Orbital inclination
inc_0 = np.radians(20) #20
#Longitude of the Ascending Node
Om_0 = np.radians(0) #0

#Stellar vector
ra = np.radians(0) #90
dec = np.radians(90)#-40

#The max distance to the other satellites in m
delta_r_max = 0.3e3

#List of perturbations: 1 = J2, 2 = Solar radiation, 3 = Drag. Leave empty list if no perturbations.
p_list = [1] #Currently just using J2

#------------------------------------------------------------------------------------------
#Calculate reference orbit, in the geocentric (ECI) frame (See Orbit module)
ECI = ECI_orbit(R_orb, delta_r_max, inc_0, Om_0, ra, dec)

#Number of orbits
n_orbits = 0.5
#Number of phases in each orbit
n_phases = 5000
#Total evaluation points
n_times = int(n_orbits*n_phases)
times = np.linspace(0,ECI.period*n_orbits,n_times) #Create list of times

"""Initialise state arrays"""
ECI_rc = np.zeros((n_times,6)) #Chief state
s_hats = np.zeros((n_times,3)) #Star vectors

#Calculate the positions of the chief and deputies in the absence of
#perturbations in both the ECI and LVLH frames
for i in range(n_times):
    chief = Chief(ECI,times[i],True)
    ECI_rc[i] = chief.state
    s_hats[i] = np.dot(chief.mat,ECI.s_hat) #Star vectors

chief_0 = Chief(ECI,0)
LVLH_drd1_0 = init_deputy(ECI,chief_0,1).to_LVLH(chief_0)
LVLH_drd2_0 = init_deputy(ECI,chief_0,2).to_LVLH(chief_0)

J2_func1 = J2_pet(LVLH_drd1_0,ECI)
J2_func2 = J2_pet(LVLH_drd2_0,ECI)

def delv_burn(r0,v0,r,v,t):
    delv_r = 1.5*(r-r0)/t
    delv_v = 1.*(v - v0)
    return delv_r + delv_v

t_burn = 40000
delv_ls = []

t_next = 0

def diff_func(t,state):
    state1 = state[:6]
    state2 = state[6:]
    #print(state1[0] + state1[3]/ECI.ang_vel)
    out1 = J2_func1(t,state1)
    out2 = J2_func2(t,state2)
    return np.concatenate((out1,out2))


def feval(funcName, *args):
    return eval(funcName)(*args)


def RKF45(func, yinit, x_range, h):
    m = len(yinit)
    n = int((x_range[-1] - x_range[0])/h)
    
    x = x_range[0]
    y = yinit
    
    # Containers for solutions
    xsol = np.empty(0)
    xsol = np.append(xsol, x)

    ysol = np.zeros((n+1,m))
    ysol[0] = y

    for i in range(n):
        print(i)
        k1 = feval(func, x, y)

        yp2 = y + k1*(h/5)

        k2 = feval(func, x+h/5, yp2)

        yp3 = y + k1*(3*h/40) + k2*(9*h/40)

        k3 = feval(func, x+(3*h/10), yp3)

        yp4 = y + k1*(3*h/10) - k2*(9*h/10) + k3*(6*h/5)

        k4 = feval(func, x+(3*h/5), yp4)

        yp5 = y - k1*(11*h/54) + k2*(5*h/2) - k3*(70*h/27) + k4*(35*h/27)

        k5 = feval(func, x+h, yp5)

        yp6 = y + k1*(1631*h/55296) + k2*(175*h/512) + k3*(575*h/13824) + k4*(44275*h/110592) + k5*(253*h/4096)

        k6 = feval(func, x+(7*h/8), yp6)

        for j in range(m):
            y[j] = y[j] + h*(37*k1[j]/378 + 250*k3[j]/621 + 125*k4[j]/594 + 512*k6[j]/1771)
            #print(y[j])

        x = x + h
        xsol = np.append(xsol, x)

        ytmp = []
        for r in range(len(y)):
            ytmp.append(y[r])    
        ysol[i+1] = ytmp  

    return [xsol, ysol]


def rungeKutta(dydx, x0, y0, x_ls, h):
    y_ls = []
    for x in x_ls:
        print(x)
        # Count number of iterations using step size or 
        # step height h 
        n = (int)((x - x0)/h)  
        # Iterate for number of iterations 
        y = y0 
        for i in range(1, n + 1): 
            "Apply Runge Kutta Formulas to find next value of y"
            k1 = h * dydx(x0, y) 
            k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1) 
            k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2) 
            k4 = h * dydx(x0 + h, y + k3) 
      
            # Update next value of y 
            y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4) 
      
            # Update next value of x 
            x0 = x0 + h 
        y_ls.append(y)
        y0 = y
        x0 = x   
    y_ls = np.array(y_ls)
    return y_ls
    
kappa1 = 0
kappa2 = 0
def rungeKutta2(dydx, x0, y0, x_ls, h):
    y_ls = []
    for x in x_ls:
        # Count number of iterations using step size or 
        # step height h 
        n = (int)((x - x0)/h)  
        # Iterate for number of iterations 
        y = y0 
        for i in range(1, n + 1):
            """
            global t_next
            global delv_ls
            if x0 > t_next:
                print(x0)
                chief = Chief(ECI,x0,True)
                s_hat = np.dot(chief.mat,ECI.s_hat)
                #print(s_hat)
                
                pos1 = y[:3]
                pos2 = y[6:9]
                vel1 = y[3:6]
                vel2 = y[9:]
                
                del_s1 = np.dot(pos1,s_hat)*s_hat
                del_s2 = np.dot(pos2,s_hat)*s_hat
                #del_sv1 = 0*np.dot(vel1,s_hat)*s_hat
                #del_sv2 = 0*np.dot(vel2,s_hat)*s_hat
                
                #delv_s1 = delv_burn(del_s1,del_sv1,0,0,t_burn)
                #delv_s2 = delv_burn(del_s2,del_sv2,0,0,t_burn)
                
                delv_s1 = kappa1/t_burn*(0-del_s1)
                delv_s2 = kappa1/t_burn*(0-del_s2)
                
                new_pos1 = pos1 - del_s1
                new_pos2 = pos2 - del_s2
                
                #new_vel1 = vel1 - del_sv1
                #new_vel2 = vel2 - del_sv2
                
                del_b = new_pos2 - new_pos1
                del_b_half = 0.5*del_b
                m0 = new_pos1 + del_b_half
                
                #del_bv = new_vel1 - new_vel2
                #del_bv_half = 0.5*del_bv
                #mv0 = (new_vel1 + del_bv_half)
                
                delv_bc = kappa2/t_burn*(m0)
                delv_bd = kappa2/t_burn*(-m0)
                
                #delv_bc = delv_burn(0,0,m0,mv0,t_burn)
                #delv_bd = delv_burn(0,0,-m0,-mv0,t_burn)
                
                vel1 += delv_bd + delv_s1
                vel2 += delv_bd + delv_s2
                
                state1 = np.concatenate((pos1,vel1))
                state2 = np.concatenate((pos2,vel2))
                
                y = np.concatenate((state1,state2))
                delv_ls.append([delv_bc,delv_s1,delv_s2])
                t_next += t_burn            
                
        
            """
            "Apply Runge Kutta Formulas to find next value of y"
            k1 = h * dydx(x0, y) 
            k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1) 
            k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2) 
            k4 = h * dydx(x0 + h, y + k3) 
      
            # Update next value of y 
            y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4) 
      
            # Update next value of x 
            x0 = x0 + h 
        y_ls.append(y)
        y0 = y
        x0 = x   
    y_ls = np.array(y_ls)
    return y_ls

state0 = np.concatenate((LVLH_drd1_0.state,LVLH_drd2_0.state))

#Tolerance and steps required for the integrator
rtol = 1e-9
atol = 1e-18
step = 10
"""
def optimize(var):

    kappa1,kappa2,t_burn = var
    global delv_ls
    delv_ls = []

    def rungeKutta3(dydx, x0, y0, x_ls, h):
        y_ls = []
        t_next = t_burn
        global delv_ls
        for x in x_ls:
            # Count number of iterations using step size or 
            # step height h 
            n = (int)((x - x0)/h)  
            # Iterate for number of iterations 
            y = y0 
            
            for i in range(1, n + 1):
            
                if x0 > t_next:
                    chief = Chief(ECI,x0,True)
                    s_hat = np.dot(chief.mat,ECI.s_hat)
                    #print(s_hat)
                    
                    pos1 = y[:3]
                    pos2 = y[6:9]
                    vel1 = y[3:6]
                    vel2 = y[9:]
                    
                    del_s1 = np.dot(pos1,s_hat)*s_hat
                    del_s2 = np.dot(pos2,s_hat)*s_hat

                    delv_s1 = kappa1/t_burn*(0-del_s1)
                    delv_s2 = kappa1/t_burn*(0-del_s2)
                    
                    new_pos1 = pos1 - del_s1
                    new_pos2 = pos2 - del_s2

                    del_b = new_pos2 - new_pos1
                    del_b_half = 0.5*del_b
                    m0 = new_pos1 + del_b_half

                    delv_bc = kappa2/t_burn*(m0)
                    delv_bd = kappa2/t_burn*(-m0)

                    
                    vel1 += delv_bd + delv_s1
                    vel2 += delv_bd + delv_s2
                    
                    state1 = np.concatenate((pos1,vel1))
                    state2 = np.concatenate((pos2,vel2))
                    
                    y = np.concatenate((state1,state2))
                    delv_ls.append([delv_bc,delv_s1,delv_s2])
                    t_next += t_burn            
                    
            
                "Apply Runge Kutta Formulas to find next value of y"
                k1 = h * dydx(x0, y) 
                k2 = h * dydx(x0 + 0.5 * h, y + 0.5 * k1) 
                k3 = h * dydx(x0 + 0.5 * h, y + 0.5 * k2) 
                k4 = h * dydx(x0 + h, y + k3) 
          
                # Update next value of y 
                y = y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4) 
          
                # Update next value of x 
                x0 = x0 + h 
            y_ls.append(y)
            y0 = y
            x0 = x   
        y_ls = np.array(y_ls)
        return y_ls
    
    X = rungeKutta3(diff_func,0,state0,times,0.1)
    
    pert_LVLH_drd1 = X[:,:6]
    pert_LVLH_drd2 = X[:,6:]
    
    total_sep = np.zeros(n_times) #Total separation

    
    for ix in range(n_times):
        #Baseline separations is simply the difference between the positions of the two deputies
        baseline_sep = np.linalg.norm(pert_LVLH_drd1[ix,:3]) - np.linalg.norm(pert_LVLH_drd2[ix,:3])
        #Component of perturbed orbit in star direction
        s_hat_drd1 = np.dot(pert_LVLH_drd1[ix,:3],s_hats[ix])
        s_hat_drd2 = np.dot(pert_LVLH_drd2[ix,:3],s_hats[ix])
        #Separation of the two deputies in the star direction
        s_hat_sep = s_hat_drd1 - s_hat_drd2
        #Sum of the separation along the star direction and the baseline direction
        total_sep[ix] = baseline_sep + s_hat_sep
        
    max_sep = np.max(np.abs(total_sep))
    
    p = [[np.linalg.norm(x) for x in delv_ls[j]] for j in range(len(delv_ls))]
    delv_max = np.sum(p)
    
    print("var = %s, max_sep = %.5f, delv = %.5f"%(var,max_sep,delv_max))
    
    cost_func = 0.01*delv_max + 0.1*max_sep
    return cost_func
    
def true_optimise():
    sol = minimize(optimize,[1,1,10])
    return sol
"""
"""
X = rungeKutta(diff_func,0,state0,times,0.1)
    
pert_LVLH_drd1 = X[:,:6]
pert_LVLH_drd2 = X[:,6:]
"""
X = RKF45("diff_func",state0,times,0.1)
pert_LVLH_drd1 = X[1][:,:6]
pert_LVLH_drd2 = X[1][:,6:]

"""
#Integrate the orbits using HCW and Perturbations D.E (Found in perturbation module)
X = solve_ivp(diff_func, [times[0],times[-1]], state0, t_eval = times, rtol = rtol, atol = atol, max_step=step)
#Check if successful integration
if not X.success:
    raise Exception("Integration failed!!!!")

#Peturbed orbits
pert_LVLH_drd1 = np.transpose(X.y)[:,:6]
pert_LVLH_drd2 = np.transpose(X.y)[:,6:]
"""
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
    baseline_sep[ix] = np.linalg.norm(pert_LVLH_drd1[ix,:3]) - np.linalg.norm(pert_LVLH_drd2[ix,:3])
    #Component of perturbed orbit in star direction
    s_hat_drd1[ix] = np.dot(pert_LVLH_drd1[ix,:3],s_hats[ix])
    s_hat_drd2[ix] = np.dot(pert_LVLH_drd2[ix,:3],s_hats[ix])
    #Baseline unit vector
    b_hat = pert_LVLH_drd1[ix,:3]/np.linalg.norm(pert_LVLH_drd1[ix,:3])
    #Component of perturbed orbit in baseline direction
    b_hat_drd1[ix] = np.dot(pert_LVLH_drd1[ix,:3],b_hat)
    b_hat_drd2[ix] = np.dot(pert_LVLH_drd2[ix,:3],b_hat)
    #Separation of the two deputies in the star direction
    s_hat_sep[ix] = s_hat_drd1[ix] - s_hat_drd2[ix]
    #Sum of the separation along the star direction and the baseline direction
    total_sep[ix] = baseline_sep[ix] + s_hat_sep[ix]

#Numerical differentiation twice - position -> acceleration
def acc(pos,times):
    vel = np.gradient(pos, times, edge_order=2)
    acc = np.gradient(vel, times, edge_order=2)
    return acc

#Accelerations - numerically integrate the position/time arrays found above
#Returns the absolute value of the acceleration in a given direction
acc_s1 = np.abs(acc(s_hat_drd1,times))
acc_s2 = np.abs(acc(s_hat_drd2,times))
acc_delta_b = np.abs(acc(baseline_sep,times))
acc_delta_s = np.abs(acc(s_hat_sep,times))
acc_total = np.abs(acc(total_sep,times))

#Maximum accelerations
max_acc_s1 = max(acc_s1)
max_acc_s2 = max(acc_s2)
max_acc_delta_b = max(acc_delta_b)
max_acc_delta_s = max(acc_delta_s)
max_acc_total = max(acc_total)

#Delta v (Integral of the absolute value of the acceleration)
delta_v_s1 = np.trapz(acc_s1)
delta_v_s2 = np.trapz(acc_s2)
delta_v_delta_b = np.trapz(acc_delta_b)
delta_v_delta_s = np.trapz(acc_delta_s)
delta_v_total = np.trapz(acc_total)

#Result array
#result[0] is the max a between deputy 1 and chief in the star direction
#result[1] is the max a between deputy 2 and chief in the star direction
#result[2] is the max a between the two deputies in the star direction (ie the difference between 0 and 1)
#result[3] is the max a in the baseline direction
#result[4] is the max total a (sum of 2 and 3; the total acceleration that needs to be corrected for)
#result[5-9] are the same, but for delta v

result = np.array([max_acc_s1,max_acc_s2,max_acc_delta_s,
                   max_acc_delta_b,max_acc_total,delta_v_s1,delta_v_s2,
                   delta_v_delta_s,delta_v_delta_b,delta_v_total])

# ---------------------------------------------------------------------- #
### PLOTTING STUFF ###

### Functions to set 3D axis aspect ratio as equal
def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

#Plot ECI Orbit
plt.figure(1)
plt.clf()
ax1 = plt.axes(projection='3d')
ax1.set_aspect('equal')
ax1.plot3D(ECI_rc[:,0],ECI_rc[:,1],ECI_rc[:,2],'b-')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('z (m)')
ax1.set_title('Orbit in ECI frame')
set_axes_equal(ax1)

#Plot perturbed LVLH orbits
plt.figure(2)
plt.clf()
ax2 = plt.axes(projection='3d')
ax2.set_aspect('equal')
ax2.plot3D(pert_LVLH_drd1[:,0],pert_LVLH_drd1[:,1],pert_LVLH_drd1[:,2],'b--')
ax2.plot3D(pert_LVLH_drd2[:,0],pert_LVLH_drd2[:,1],pert_LVLH_drd2[:,2],'c--')
ax2.set_xlabel('r (m)')
ax2.set_ylabel('v (m)')
ax2.set_zlabel('h (m)')
ax2.set_title('Orbit in LVLH frame')
set_axes_equal(ax2)

#Plot separation along the star direction
plt.figure(3)
plt.clf()
#plt.plot(times,s_hat_drd1,"b-",label="SCHWEIGHART Deputy 1, s direction")
#plt.plot(times,s_hat_drd2,"g-",label="SCHWEIGHART Deputy 2, s direction")
#plt.plot(times,s_hat_sep,"r-",label="SCHWEIGHART Separation, s direction")
plt.plot(times,s_hat_drd1,"b-",label="Deputy 1, s direction")
plt.plot(times,s_hat_drd2,"g-",label="Deputy 2, s direction")
plt.plot(times,s_hat_sep,"r-",label="Separation, s direction")
plt.xlabel("Times(s)")
plt.ylabel("Separation(m)")
plt.title('Separations against time due to perturbations')
plt.legend()

#Plot separation along the baseline direction
plt.figure(4)
plt.clf()
plt.plot(times,baseline_sep,"y-",label="Separation, baseline direction")
#plt.plot(times,total_sep,"c-",label="Total direction")
plt.xlabel("Times(s)")
plt.ylabel("Separation(m)")
plt.title('Separations against time due to perturbations')
plt.legend()

#Plot separation in the baseline frame
plt.figure(5)
plt.clf()

points1 = np.array([b_hat_drd1, s_hat_drd1]).T.reshape(-1, 1, 2)
points2 = np.array([b_hat_drd2, s_hat_drd2]).T.reshape(-1, 1, 2)
segments1 = np.concatenate([points1[:-1], points1[1:]], axis=1)
segments2 = np.concatenate([points2[:-1], points2[1:]], axis=1)
norm = plt.Normalize(times.min(), times.max())
ax = plt.gca()
lc1 = LineCollection(segments1, cmap='YlOrRd', norm=norm)
lc1.set_array(times)
lc1.set_linewidth(2)
ax.add_collection(lc1)
lc2 = LineCollection(segments2, cmap='YlGnBu', norm=norm)
lc2.set_array(times)
lc2.set_linewidth(2)
ax.add_collection(lc2)
space_f = 1.2
plt.xlim(np.min(space_f*np.minimum(b_hat_drd1,b_hat_drd2)), np.max(space_f*np.maximum(b_hat_drd1,b_hat_drd2)))
plt.ylim(np.min(space_f*np.minimum(s_hat_drd1,s_hat_drd2)), np.max(space_f*np.maximum(s_hat_drd1,s_hat_drd2)))
plt.xlabel("Baseline direction (m)")
plt.ylabel("Star direction (m)")
plt.title("Position of deputies due to \n perturbations in Baseline frame")

cbar = plt.colorbar(lc1)
plt.colorbar(lc2)
#cbar.set_label('Time (Schweighart) (s)', rotation=270, labelpad = 15)
cbar.set_label('Time (s)', rotation=270, labelpad = 15)

