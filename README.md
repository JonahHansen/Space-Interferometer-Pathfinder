# Astrophysical Space-Interferometer Pathfinder (ASP)
Jonah Honours Project on Space Interferometry and Orbital Dynamics... This repo is a bit of a mess

### Main files (Scripts):

* plotter - plots the orbit in 3D and animates it
* perturbation_script - script to calculate (integrate), as well as plot the orbit with given perturbations (Schweighart/HCW method)
* pert_test2 - Script comparing different J2 perturbation methods (none really work...)
* obs_script - Script to test observability functions, and calculate observability plots
* obs_time_read - script to read the numpy arrays produced by obs_script
* uv_script - Script to calculate the UV plane over a period of time and plot it
* delv_script - Calculates the delta v required through a full orbit simulation (WIP)



### Modules

* quaternions.py: helper functions for quaternion rotation
* orbits - contains the reference orbit class, calculates and holds the relevant orbit parameters, as well as helper functions to calculate the state vectors of the satellites in the different frames
* Schweighart_J2 - module that uses the J2 perturbation function given by Schweighart 2002. Uses the absolute numerical implementation.
* Schweighart_J2_solved - module that uses the J2 perturbation function given by Schweighart 2002. Uses the absolute analytical solution. Currently preferred.
* Old_perturbations - module containing the time varying LVLH transition matrix method to calculate perturbation. Probably broken.
* Reconfiguration - Calculates the delta-v requirement to reconfigure an orbit (Not a great module, may be currently broken)
* Observability - Functions to determine whether the star is viewable or not

### Junk:
 Many, many old versions of code. Later versions of Old are relatively newer. Use to jumpstart other scripts
