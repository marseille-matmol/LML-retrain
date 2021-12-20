import os
import numpy as np
import pandas as pd

from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from ase.neighborlist import neighbor_list
from ase.optimize import FIRE, LBFGSLineSearch
from ase.neb import NEB
from ase.geometry.geometry import get_distances
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

import matscipy.dislocation as sd

def make_ML_calc(potentails_folder="../potentials",
                 pot_name="W_milady"):

    pot_files = [f"{pot_name}.snapcoeff", f"{pot_name}.snapparam"]
    pot_paths = [os.path.join(potentails_folder, pot_file) for pot_file in pot_files]
    pot_path = " ".join(pot_paths)

    ML_calc = LAMMPSlib(lmpcmds=["pair_style snap",
                                 "pair_coeff * * %s W" % pot_path],
                        atom_types={'W': 1}, keep_alive=True)
    return ML_calc


def Vac_migration(calculator,
                  fmax_relaxation=1.0e-3,
                  fmax_neb=1.0e-2,
                  n_knots=11):

    alat, _, _, _ = sd.get_elastic_constants(calculator=calculator)

    W_bulk = bulk("W", a=alat, cubic="True")
    W_bulk *= 5

    W_bulk.calc = calculator
    #opt = FIRE(W_bulk)
    opt = LBFGSLineSearch(W_bulk)

    opt.run(fmax=fmax_relaxation)

    # place vacancy near centre of cell
    D, D_len = get_distances(np.diag(W_bulk.cell) / 2,
                             W_bulk.positions,
                             W_bulk.cell, W_bulk.pbc)

    vac_index = D_len.argmin()
    vac_pos = W_bulk.positions[vac_index]
    ini_vac = W_bulk.copy()
    del ini_vac[vac_index]

    # identify two opposing nearest neighbours of the vacancy
    D, D_len = get_distances(vac_pos,
                             ini_vac.positions,
                             ini_vac.cell, ini_vac.pbc)
    D = D[0, :]
    D_len = D_len[0, :]

    nn_mask = np.abs(D_len - D_len.min()) < 1e-8
    i1 = nn_mask.nonzero()[0][0]
    i2 = ((D + D[i1])**2).sum(axis=1).argmin()

    print(f'vac_index={vac_index} i1={i1} i2={i2} '
          f'distance={ini_vac.get_distance(i1, i2, mic=True)}')

    fin_vac = ini_vac.copy()
    fin_vac.positions[i1] = vac_pos

    for atoms in ini_vac, fin_vac:
        atoms.calc = calculator
        # opt = FIRE(atoms)
        opt = LBFGSLineSearch(atoms)
        opt.run(fmax=fmax_relaxation)

    np.allclose(ini_vac.get_potential_energy(), fin_vac.get_potential_energy())

    print(ini_vac.get_potential_energy(), fin_vac.get_potential_energy())

    vacancy_formation_energy = ini_vac.get_potential_energy() - W_bulk.get_potential_energy() * len(ini_vac) / len(W_bulk)

    print(f"Vacancy formation energy: {vacancy_formation_energy:.3f} eV")


    images = [ini_vac] + \
             [ini_vac.copy() for i in range(n_knots)] + \
             [fin_vac]


    for image in images:
        image.calc = calculator

    vac_NEB = NEB(images, allow_shared_calculator=True)
    vac_NEB.interpolate()
    #opt = LBFGSLineSearch(vac_NEB)
    #opt.run(fmax=fmax_neb, steps=2)
    opt = FIRE(vac_NEB)
    opt.run(fmax=fmax_neb)
    return vac_NEB, vacancy_formation_energy


if __name__ == '__main__':

    fmax_relaxation = 1.0e-3
    fmax_neb = 0.02
    n_knots = 11

    dataframe = pd.DataFrame()

    xyz_folder = "../data_files/ML_vac_migration_images/"

    if not os.path.exists(xyz_folder):
        os.mkdir(xyz_folder)

    pot_folder = "../potentials"
    pot_name = "milady_lml"

    calculator = make_ML_calc(pot_name=pot_name,
                              potentails_folder=pot_folder)
    saving_fname = f"{pot_name}_vac_mig_images.xyz"

    vac_neb, vac_formen = Vac_migration(calculator, n_knots=n_knots,
                                        fmax_relaxation=fmax_relaxation,
                                        fmax_neb=fmax_neb)

    dataframe = dataframe.append({"pot_name": pot_name,
                                  "vac_formen": vac_formen},
                                  ignore_index=True)

    dataframe.to_csv(f"../data_files/vac_formation_energies.csv")

    for image in vac_neb.images:
        energy = image.get_potential_energy()
        forces = image.get_forces()
        image.calc = SinglePointCalculator(image, energy=energy, forces=forces)

    write(xyz_folder + saving_fname, vac_neb.images)

    pot_folder = "../potentials/Lam0_500_LamS_1/"
    label = "qSNAP"
    pot_names = [fn.split(".")[0] for fn in os.listdir(pot_folder) if label in fn and "snapcoef" in fn]
    print(f"Found {len(pot_names)} potentials:")
    print(pot_names)

    for pot_name in pot_names:

        pot_files = [f"{pot_name}.snapcoeff", f"{pot_name}.snapparam"]
        calculator = make_ML_calc(pot_name=pot_name,
                                  potentails_folder=pot_folder)

        saving_fname = f"{pot_name}_vac_mig_images.xyz"

        vac_neb, vac_formen = Vac_migration(calculator, n_knots=n_knots,
                                            fmax_relaxation=fmax_relaxation,
                                            fmax_neb=fmax_neb)

        dataframe = dataframe.append({"pot_name": pot_name,
                                      "vac_formen": vac_formen},
                                      ignore_index=True)

        dataframe.to_csv(f"../data_files/vac_formation_energies.csv")

        for image in vac_neb.images:
            energy = image.get_potential_energy()
            forces = image.get_forces()
            image.calc = SinglePointCalculator(image, energy=energy, forces=forces)

        write(xyz_folder + saving_fname, vac_neb.images)
