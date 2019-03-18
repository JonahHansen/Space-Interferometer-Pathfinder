# OrbitDynamics
Jonah Honours Project on Orbital Dynamics

### Main Files:

* plotter - plots the orbit in 3D and animates it
* quaternions.py: helper functions for quaternion rotation
* orbits - contains the orbit class, calculates and holds the relevant orbit parameters, as well as helper functions to calculate the state vectors of the satellites in the different frames
* perturbation_script - script to calculate (and integrate) the orbit with given perturbation (currently just J2)
* perturbation_script_noJ2 - script to calculate (and integrate) the orbit with no perturbation, but using the same base script as the one above
* Schweighart_J2 - module that uses the J2 perturbation function given by Schweighart 2002. Doesn't currently work

### Junk:
* plotter1 - Mike's original code
* plotter2 - Messed around with the rotations, though it doesn't keep phase
* plotter3 - Using the intersecting plane idea to find the deputy orbits
* plotter4 - Numerically finding the orbit using the intersecting plane
* plotter5 - Same as the original, but using quaternions instead of matrices (same result, but slower and easier to use)

### Old:
Old versions of (semi) working code. Not quite as junky as the stuff in junk.
