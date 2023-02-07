LML_retrain
===========

This repository contains implementation of constrained retraining procedure for linear machine learning potentials as described in 
[*Calculation of dislocation binding to helium-vacancy defects in tungsten using hybrid ab initio-machine learning methods*](https://doi.org/10.1016/j.actamat.2023.118734), or an earlier preprint on [*arXiv*](https://arxiv.org/abs/2111.11262)


For basic retraining only [numpy](https://numpy.org/) and [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orth.html) are required. 

- Evaluating atomic structures and generating descriptor vectors requires [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/) on top of the `LAMMPS` [python module](https://docs.lammps.org/Python_module.html). 

- `LAMMPS` should be compiled with [`ML-SNAP` package](https://docs.lammps.org/pair_snap.html#restrictions) which is not included by default, i.e. with a traditional make for `LAMMPS`
```
  cd /path/to/lammps/src
  make yes-ML-SNAP
  make yes-[OTHER PACKAGES]
  make mpi mode=shlib
```

- Script for calculating dislocation glide barriers depends on [matscipy](https://github.com/libAtoms/matscipy) dislocation module. All of these python packages can be easily installed via `pip` or other package managers.
