import os

import numpy as np
import pandas as pd

import matscipy.dislocation as sd

#from mpi4py import MPI
from ase.calculators.lammpslib import LAMMPSlib
from ase.optimize import FIRE
from ase.neb import NEB, NEBOptimizer
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write, read
from ase.constraints import FixAtoms


def add_fix_atoms_to_image(image):

    if "fix_mask" in image.arrays:

        if len(image.constraints) == 0:

            print("Adding fixed atoms constraint")

            fix_mask = image.get_array("fix_mask")
            fix_atoms = FixAtoms(mask=fix_mask)
            image.set_constraint(fix_atoms)

        else:
            print("Constraints list is not zero")
    else:
        raise RuntimeError("No fixmask array to add fixed atoms")

    return None


def glide_NEB_relaxation(disloc_ini=None, disloc_fin=None, images=None,
                         fmax_relaxation=1.0e-3,
                         fmax_neb=5.0e-3, method="aseneb",
                         n_knots=7, apply_constraint=False):

    if disloc_ini is not None and disloc_fin is not None:
        for disloc in (disloc_ini, disloc_fin):
            opt = FIRE(disloc)
            opt.run(fmax=fmax_relaxation)

    elif images is not None:
        disloc_ini = images[0]
        disloc_fin = images[-1]

    try:
        np.testing.assert_almost_equal(disloc_ini.get_potential_energy(),
                                       disloc_fin.get_potential_energy(),
                                       decimal=3)
    except AssertionError as e:
        print(e)

    if images is None:

        images = [disloc_ini] + \
                 [disloc_ini.copy() for i in range(n_knots)] + \
                 [disloc_fin]

        calculator = disloc_ini.calc

        for image in images:
            image.calc = calculator

    if method == "spline":
        glide_neb = NEB(images, allow_shared_calculator=True,
                        method=method, precon="Exp")
    else:
        glide_neb = NEB(images, allow_shared_calculator=True,
                        method=method)
    # by default interpolate applies constraints while setting new atoms positions
    if images is None:
        glide_neb.interpolate(apply_constraint=apply_constraint)
        opt = FIRE(glide_neb)

    else:
        opt = FIRE(glide_neb)

    opt.run(fmax=fmax_neb, steps=250)

    return glide_neb


if __name__ == '__main__':

    fmax_relaxation = 1.0e-3
    fmax_neb = 0.025
    n_knots = 11

    work_folder = os.environ["WORK"]
    pot_folder = os.path.join(work_folder, "gits/MeMoPAS/refit_potential/potentials")

    label = "qSNAP_DFT"
    pot_names = [fn.split(".")[0] for fn in os.listdir(pot_folder) if label in fn and "snapcoef" in fn]
    print(f"Found {len(pot_names)} potentials:")
    print(pot_names)

    dataframe = pd.DataFrame()

    for pot_name in pot_names:

        pot_files = [f"{pot_name}.snapcoeff", f"{pot_name}.snapparam"]
        pot_paths = [os.path.join(pot_folder, pot_file) for pot_file in pot_files]
        pot_path = " ".join(pot_paths)

        lammps = LAMMPSlib(lmpcmds=["pair_style snap",
                           "pair_coeff * * %s W" % pot_path],
                           atom_types={'W': 1}, keep_alive=True)#  ,log_file="lammps.log")

        elastic_params = sd.get_elastic_constants(calculator=lammps, delta=5.0e-3)

        print(elastic_params)
        dataframe = dataframe.append({"pot_name": pot_name,
                                      "a0": elastic_params[0],
                                      "C11": elastic_params[1],
                                      "C12": elastic_params[2],
                                      "C44": elastic_params[3]},
                                      ignore_index=True)

        dataframe.to_csv(f"elastic_params.csv")

        dislocations = {"edge100110": sd.BCCEdge100110Dislocation(*elastic_params),
                        "edge100": sd.BCCEdge100Dislocation(*elastic_params),
                        "screw": sd.BCCScrew111Dislocation(*elastic_params)}
                        #"mixed": sd.BCCMixed111Dislocation(*elastic_params),
                        #"edge111": sd.BCCEdge111Dislocation(*elastic_params)}

        radii = {"edge100110": 85,
                 "edge100": 85,
                 "screw": 60,
                 "mixed": 85,
                 "edge111": 85}

        pot_glide_results = {}

        for dislo_name, dislocation in dislocations.items():
            print(f"Running {dislo_name} with: {pot_name}")

            saving_fname = f"{pot_name}_glide_{dislo_name}_images.xyz"

            if os.path.exists(saving_fname):
                print(f"Reading from {saving_fname}")
                saved_images = read(saving_fname, index=":")

                for image in saved_images:
                    add_fix_atoms_to_image(image)
                    image.calc = lammps

                glide_neb = glide_NEB_relaxation(images=saved_images,
                                                 fmax_neb=fmax_neb,
                                                 method="spline",
                                                 fmax_relaxation=fmax_relaxation)

            else:
                bulk, disloc_ini, disloc_fin = dislocation.build_glide_configurations(radius=radii[dislo_name])

                disloc_ini.calc = lammps
                disloc_fin.calc = lammps
                glide_neb = glide_NEB_relaxation(disloc_ini, disloc_fin,
                                                 fmax_neb=fmax_neb,
                                                 fmax_relaxation=fmax_relaxation)

            for image in glide_neb.images:
                energy = image.get_potential_energy()
                forces = image.get_forces()
                image.calc = SinglePointCalculator(image, energy=energy, forces=forces)

            write(saving_fname, glide_neb.images)
