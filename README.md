# OrbitDynamics
Jonah Honours Project on Orbital Dynamics

### Main files (Scripts):

* plotter - plots the orbit in 3D and animates it
* perturbation_script - script to calculate (integrate), as well as plot the orbit with given perturbations
* perturbations_multiprocessing - calculates acceleration due to perturbations for many orbits with multiprocessing
* perturbation_script_Schweighart - same as perturbation script, but uses the Schweighart formulation. Preferred script.
* Acc_plot - Plots the data from perturbations_multiprocessing
* Obs_script - Script to test observability functions, plots u-v plane and calculates observability
* Obs_multi - multiprocessing version of obs_script. Does not calculate the u-v plane
* Read_observability - Read (and plot) data from obs_multi

### Modules

* quaternions.py: helper functions for quaternion rotation
* orbits - contains the orbit class, calculates and holds the relevant orbit parameters, as well as helper functions to calculate the state vectors of the satellites in the different frames
* perturbations - module that calculates the acceleration to propagate the orbit, as well as the perturbation functions
* Schweighart_J2 - module that uses the J2 perturbation function given by Schweighart 2002. Now preferred over perturbations
* Reconfiguration - Calculates the delta-v requirement to reconfigure an orbit
* Observability - Functions to determine whether the star is viewable or not

### Junk:
* plotter1 - Mike's original code
* plotter2 - Messed around with the rotations, though it doesn't keep phase
* plotter3 - Using the intersecting plane idea to find the deputy orbits
* plotter4 - Numerically finding the orbit using the intersecting plane
* plotter5 - Same as the original, but using quaternions instead of matrices (same result, but slower and easier to use)

### Old:
Old versions of (semi) working code. Not quite as junky as the stuff in junk.
