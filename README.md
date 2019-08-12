# Astrophysical Space-Interferometer Pathfinder (ASP)
Jonah Honours Project on Space Interferometry and Orbital Dynamics

### Main files (Scripts):

* plotter - plots the orbit in 3D and animates it
* perturbation_script - script to calculate (integrate), as well as plot the orbit with given perturbations (Schweighart Numerical method)
* multi_pert- calculates acceleration due to perturbations for many orbits with multiprocessing
* perturbation_script_sol - same as perturbation script, but uses the Analytical Schweighart formulation. Preferred script.
* Acc_plot - Plots the data from multi_pert
* Obs_script - Script to test observability functions, plots u-v plane and calculates observability
* Obs_multi - multiprocessing version of obs_script. Does not calculate the u-v plane
* Read_observability - Read (and plot) data from obs_multi
* delv3 - Currently unworking full orbit simulation. Needs a lot of work.

### Modules

* quaternions.py: helper functions for quaternion rotation
* orbits - contains the reference orbit class, calculates and holds the relevant orbit parameters, as well as helper functions to calculate the state vectors of the satellites in the different frames
* Schweighart_J2 - module that uses the J2 perturbation function given by Schweighart 2002. Uses the absolute numerical implementation.
* Schweighart_J2_rel - module that uses the J2 perturbation function given by Schweighart 2002. Uses the relative numerical implementation.
* Schweighart_J2_solved - module that uses the J2 perturbation function given by Schweighart 2002. Uses the absolute analytical solution. Currently preffered.
* Reconfiguration - Calculates the delta-v requirement to reconfigure an orbit (Not a great module, may be currently broken)
* Observability - Functions to determine whether the star is viewable or not

### Junk:
 Many old versions of code. Later versions of Old are relatively newer. Use to jumpstart other scripts
