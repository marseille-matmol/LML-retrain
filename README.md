LML_retrain
===========

This repository contains implementation of constrained retraining procedure for linear machine learning potentials as described in [*Synergistic coupling in ab initio-machine learning simulations of dislocations*](https://arxiv.org/abs/2111.11262)


For basic retraining only [numpy](https://numpy.org/) and [scipy.linalg.orth](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orth.html) installation is required. Evaluating atomic structures and generating descriptor vectors requires [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/) on top of [LAMMPS python module](https://docs.lammps.org/Python_module.html). Note that LAMMPS should be compiled with [ML-SNAP package](https://docs.lammps.org/pair_snap.html#restrictions) which is not included by default. Script for calculating dislocation glide barriers depends on [matscipy](https://github.com/libAtoms/matscipy) dislocation module. All of these python packages can be easily installed via `pip` or other package managers.
