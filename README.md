# OrbitDynamics
Jonah Honours Project on Orbital Dynamics

### Main Files:

* plotter - plots the orbit in 3D and animates it
* quaternions.py: helper functions for quaternion rotation
* orbits - contains the orbit class, calculates and holds the relevant orbit parameters, as well as helper functions to calculate the state vectors of the satellites in the different frames
* perturbations - module that calculates the acceleration to propagate the orbit, as well as the perturbation functions
* perturbation_script - script to calculate (integrate), as well as plot the orbit with given perturbations
* perturbations_multiprocessing - calculates acceleration due to perturbations for many orbits with multiprocessing
* Schweighart_J2 - module that uses the J2 perturbation function given by Schweighart 2002. Doesn't currently work
* Acc_plot - Plots the data from perturbations_multiprocessing

### Junk:
* plotter1 - Mike's original code
* plotter2 - Messed around with the rotations, though it doesn't keep phase
* plotter3 - Using the intersecting plane idea to find the deputy orbits
* plotter4 - Numerically finding the orbit using the intersecting plane
* plotter5 - Same as the original, but using quaternions instead of matrices (same result, but slower and easier to use)

### Old:
Old versions of (semi) working code. Not quite as junky as the stuff in junk.
