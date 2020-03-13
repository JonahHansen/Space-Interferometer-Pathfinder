# Astrophysical Space-Interferometer Pathfinder (ASP)
Jonah Honours Project on Space Interferometry and Orbital Dynamics... This repo is less of a mess than it was...

### Main files (Scripts):

* plotter - plots the orbit in 3D and animates it
* LVLH_perturbation_script - script to calculate and plot the orbit with J2 perturbation in the LVLH frame (Analytical Schweighart/HCW method)
* ECI_perturbation_script - script to calculate and plot the orbit with J2 perturbation in the ECI frame (Numerical method)
* pert_test2 - Script comparing different J2 perturbation methods (none really work... and is honestly a mess of a script)
* obs_script - Script to test observability functions, and calculate observability plot of a slice of RA/DEC over time
* obs_multi - Calculate observability plot of a map of RA and DEC. Requires huge amount of memory
* uv_script - Script to calculate the UV plane over a period of time and plot it
* delv_script - Calculates the delta v required during multiple orbits through a full orbit simulation
* reconfig_script - Script to calculate the delta v required in reconfiguring the orbit

### Modules

* quaternions - helper functions for quaternion rotation
* orbits - contains the reference orbit class, which calculates and holds the relevant orbit parameters. Also has the Satellite class, used to help calculate the state vectors of the satellites in the different frames. Finally holds the functions that initialise the chief and deputy satellites.
* Numerical_LVLH_motion - module that uses the numerical J2 perturbation function given by Schweighart 2002.
* Analytical_LVLH_motion - module that uses the analytical J2 perturbation function given by Schweighart 2002. Can also reduce down to the analytical HCW equations.
* old_perturbations - module containing the time varying LVLH transition matrix method to calculate perturbation. Almost certainly broken.
* reconfiguration - Calculates the delta-v requirement to reconfigure an orbit
* observability - Functions to determine whether the star is viewable or not
