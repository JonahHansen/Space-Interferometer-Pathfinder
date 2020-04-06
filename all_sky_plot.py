"""
Create 2 random sets of targets...
1) Targets randomly distributed over the sphere. i.e. f(dec) proportional to 
dec. Cumulative distribution is 0.5*(1+sin(delta)), and inverse cumulative 
distribution is np.degrees(np.arcsin(2*F-1)) 
2) Evenly distributed stars within 10 degrees of the Galactic plane. 

In either case there is a travelling salesman problem! There is an additional restriction
that the observations have to be made within 60 degrees of antisolar.

A: Start with the random targets, and initially, observe the closest not-yet observed, 
observable target on each day. 
B: Consider swapping an existing target with another existing or reserve target
(situation is treated differently)
C: Iterate

All sky plot:

As an example, Figure~\ref{figAllSky} shows an observing sequence of 100 objects over 
a year, selected from a target list of 150 objects uniformly distributed in the sky. The
target list included objects in the unobservable ecliptic poles, and used an anti-sun angle 
of 60 degrees. The chosen path for this 
travelling salesman problem was found via simulated annealing. A mean slew distance of 3.4 
degrees per day is needed in this example, corresponding to an average of 12.5 degrees 
between targets. Increasing the number of targets 
to 360 increases the slew per day to 8.6 degrees, while restricting the targets to within
15 degrees of the Galactic plane reduces the mean slew distance to .

"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

plt.ion()

N_tgt = int(100*1.5)
N_obs = 100
N_iter = 500000
mar21_angle = (21+28+31)/365.*2*np.pi
antisun_limit = np.radians(60)
Ts = [0.3,0.1,.03,.01]
#Do we want a simulation in Galactic coordinates?
galactic=True

#-----------------

antisun_ras = 2*np.pi*(np.arange(N_obs)/N_obs)
#When the sun is at an RA of 12 hours, it is moving negative. At this point,
#the antisun is at an RA of 0 hours and moving positive.
antisun_decs = np.radians(23.5)*np.sin(antisun_ras)

def sphere_dist(lat1, lat2, dlong):
    if dlong==0:
        if lat1==lat2:
            return 0
    return np.arccos(np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(dlong))

#Uniformly distributed...
#Create some random RAs and DECs, but SORT them by RA.
ras = np.sort(np.random.random(N_tgt)*2*np.pi)
decs = np.arcsin(2*np.random.random(N_tgt)-1)

#Galactic plant only...
glong = np.random.random(N_tgt)*2*np.pi
glat = np.arcsin(np.sin(np.radians(15)) * (2*np.random.random(N_tgt)-1) )

#Now convert to RA and Dec
c = SkyCoord(l=glong*u.rad, b=glat*u.rad, frame='galactic')
c = c.transform_to('icrs')
ras = [cc.ra.to(u.rad).value for cc in c]
decs = [cc.dec.to(u.rad).value for cc in c]
sorted = np.argsort(ras)
ras = np.array(ras)[sorted]
decs = np.array(decs)[sorted]


#Now run through and create an initia list.
obs_list = [0]
for i in range(1,N_obs):
    j=0
    while len(obs_list) <= i:
        j += 1
        if j==N_tgt:
            raise UserWarning("No valid target!")
        if j in obs_list:
            continue
        if sphere_dist(antisun_decs[i], decs[j], antisun_ras[i]-ras[j]) < antisun_limit:
            obs_list += [j]
        
#Now that we've created a list, lets see how far we have to go in-between observations.
step_dist = np.zeros(N_obs-1)
for i in range(N_obs-1):
    step_dist[i] = sphere_dist(decs[obs_list[i]],decs[obs_list[i+1]],ras[obs_list[i+1]]-ras[obs_list[i]])

#Now lets randomly swap, accepting the swap if better.
plt.figure(1)
plt.clf()
plt.plot(np.degrees(ras[obs_list]), np.degrees(decs[obs_list]))
plt.plot(np.degrees(ras[obs_list]), np.degrees(decs[obs_list]),'o')  
obs_list_init = obs_list.copy()
step_dist_init = step_dist.copy() 
ks = []
dist_sums = [np.sum(step_dist)]
for T in Ts:
    for k in range(N_iter):
        i = int(np.random.random()*N_obs)
        j = int(np.random.random()*N_tgt)

        #Lets try to put target j into observation i. We first see if it is observable.
        if sphere_dist(antisun_decs[i], decs[j], antisun_ras[i]-ras[j]) > antisun_limit:
            continue
        #It doesn't make sense to swap withourselves...
        if obs_list[i]==j:
            continue    
        
        #Temporary...
        if (i>0) and (obs_list[i-1]==j):
            continue 
        if (i<N_obs-1) and (obs_list[i+1]==j):
            continue 
    
        #Next, lets compute the new travelling distances. These are:
        #Before distance for observation i, after distance for observation i
        #Before distance for observation (tgt j), after distance for observation (tgt j)
        old_dists = np.array([0.,0,0,0])
        new_dists = np.array([0.,0,0,0])
        if i>0:
            old_dists[0] = step_dist[i-1]
            new_dists[0] = sphere_dist(decs[obs_list[i-1]],decs[j],ras[j]-ras[obs_list[i-1]])
        if i<N_obs-1:
            old_dists[1] = step_dist[i]
            new_dists[1] = sphere_dist(decs[obs_list[i+1]],decs[j],ras[j]-ras[obs_list[i+1]])
        
        #Now see if target j is in the list and it has to be replaced with target obs_list[i]
        i2=-1
        if j in obs_list:
            i2 = obs_list.index(j)
            j2 = obs_list[i]
            if i2>0:
                old_dists[2] = step_dist[i2-1]
                new_dists[2] = sphere_dist(decs[obs_list[i2-1]],decs[j2],ras[j2]-ras[obs_list[i2-1]])
            if i2<N_obs-1:
                old_dists[3] = step_dist[i2]
                new_dists[3] = sphere_dist(decs[obs_list[i2+1]],decs[j2],ras[j2]-ras[obs_list[i2+1]])
    
        #Now see if the new list is better. WARNING: Check this for neighboring swap.
        if np.exp( (np.sum(old_dists)-np.sum(new_dists))/T ) > np.random.random():
            obs_list[i]=j
            if i>0:
                step_dist[i-1] = new_dists[0]
            if i<N_obs-1:
                step_dist[i] = new_dists[1]
            if i2>=0:
                obs_list[i2]=j2
                if i2>0:
                    step_dist[i2-1] = new_dists[2]
                if i2<N_obs-1:
                    step_dist[i2] = new_dists[3]
            ks += [k]
            dist_sums += [dist_sums[-1] + np.sum(new_dists) - np.sum(old_dists)]

#Now lets intelligently re-arrange the RAs...
ras_plot = ras[obs_list]
ras_plot[:N_obs//3] = ((ras_plot[:N_obs//3] + np.pi) % (2*np.pi)) - np.pi
ras_plot[2*N_obs//3:] = ((ras_plot[2*N_obs//3:] - np.pi) % (2*np.pi)) + np.pi


plt.figure(4)
plt.clf()
plt.plot(np.degrees(ras_plot), np.degrees(decs[obs_list]))    
plt.plot(np.degrees(ras_plot), np.degrees(decs[obs_list]),'.')    
plt.plot(np.degrees(ras_plot[[0,-1]]), np.degrees(decs[[obs_list[0],obs_list[-1]]]),'rx')    
plt.xlabel('RA (degs)')
plt.ylabel('Dec (degs)')
plt.tight_layout()

plt.figure(2)
plt.clf()
plt.plot(dist_sums)

plt.figure(3)
plt.clf()

plt.plot(ras_plot)
plt.plot(decs[obs_list])


#Let's check that nothing went wrong!        
step_dist_check = np.zeros(N_obs-1)
for i in range(N_obs-1):
    step_dist_check[i] = sphere_dist(decs[obs_list[i]],decs[obs_list[i+1]],ras[obs_list[i+1]]-ras[obs_list[i]])
if (np.sum(step_dist_check) != np.sum(step_dist)):
    raise UserWarning

#Now let's find the 
print("Initial slew total (degrees): {:.1f}".format(np.degrees(np.sum(step_dist_init))))
print("Final slew total (degrees): {:.1f}".format(np.degrees(np.sum(step_dist))))
print("Average slew between targets (degrees): {:.1f}".format(np.degrees(np.sum(step_dist))/N_obs))
print("Average slew per day (degrees): {:.1f}".format(np.degrees(np.sum(step_dist))/365))