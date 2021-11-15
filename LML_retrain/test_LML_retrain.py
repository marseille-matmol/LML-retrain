import pytest
import os
import numpy as np
from LML_retrain import LMLPotential


@pytest.fixture
def LML_W():
    return LMLPotential(filename="../potentials/milady_lml")

def test_load_lammps_snap_files(LML_W):
    assert LML_W.symbol == "W"
    assert LML_W.cutoff == 4.7
    assert LML_W.twojmax == 8
    # we always create the "quadratic" form from the initial linear
    assert LML_W.N_quad_desc == 56 + 56 * (56 - 1) // 2
    assert LML_W.theta_0_quad.size == LML_W.N_quad_desc

    target_thetha = np.array([-5.30602553e+00, 2.58268673e-02, 2.22488517e-02,
                              5.19196477e-02, -1.52705890e-02, 2.78235882e-01,
                              6.57811745e-02, 3.42063325e-04, -6.36810967e-03,
                              1.27078517e-01, 1.00290581e-01, 5.17892447e-02,
                              9.89949157e-02, 2.59816794e-03, -2.03236887e-02,
                              2.97954006e-02, -6.88088984e-03, 6.77710152e-02,
                              8.99507975e-02, 9.52308804e-03, 3.62687191e-02,
                              9.42854978e-04, 9.93682714e-03, -1.64980817e-03,
                              6.76059507e-02, 1.07769098e-01, 7.03231455e-02,
                              6.13290480e-02, -2.76847356e-02, 3.60451325e-02,
                              5.04902680e-02, 6.77269526e-03, -1.03409518e-02,
                              -2.03685811e-02, 2.38929115e-02, 1.01190206e-01,
                              2.91536472e-03, 1.14599124e-02, 5.58398668e-02,
                              6.68085416e-03, 3.03867864e-02, 2.07693292e-02,
                              1.25716103e-02, 6.61923683e-04, 7.36135422e-03,
                              4.05081803e-02, -6.39791620e-03, 1.31851771e-03,
                              -1.59333592e-04, -2.34161438e-03, -2.51753056e-03,
                              -8.11964183e-05, 9.44467217e-03, 1.73103269e-02,
                              -5.65618321e-03, 7.66290632e-03])

    # the resulting "quadratic" potential has only
    # non zeros components from the linear one
    np.testing.assert_allclose(LML_W.linear_theta, target_thetha)
    np.testing.assert_allclose(LML_W.theta_0_quad[:56], target_thetha)
    np.testing.assert_allclose(LML_W.theta_0_quad[56:],
                               np.zeros_like(LML_W.theta_0_quad[56:]))


def test_make_lammps_snap_files(LML_W):
    LML_W.make_lammps_snap_files("test_files", write_linear=True)

    # quadratic is false to stop doubling the dimensions
    read_LML_W = LMLPotential("test_files")

    np.testing.assert_allclose(read_LML_W.theta_0_quad,
                               LML_W.theta_0_quad)

    np.testing.assert_allclose(read_LML_W.linear_theta,
                               LML_W.linear_theta)

    assert read_LML_W.cutoff == LML_W.cutoff
    assert read_LML_W.twojmax == LML_W.twojmax
    assert read_LML_W.N_quad_desc == LML_W.N_quad_desc
    assert read_LML_W.N_linear_desc == LML_W.N_linear_desc

    os.remove("test_files.snapcoeff")
    os.remove("test_files.snapparam")

def test_get_D(LML_W):

    from ase.build import bulk
    W = bulk("W", a=3.18551, cubic=True)
    W_vac = W * 4
    # del W_vac[64]

    D = LML_W.get_D(W_vac, linear=True)

    ML_calc = LML_W.make_lammpslib_calc()
    W_vac.calc = ML_calc
    energy = W_vac.get_potential_energy()

    np.testing.assert_almost_equal((D @ LML_W.linear_theta).sum(), energy)

    D_quad = LML_W.get_D(W_vac, linear=False)
    np.testing.assert_almost_equal((D_quad @ LML_W.theta_0_quad).sum(), energy)


def test_get_dD(LML_W):

    from ase.build import bulk
    W = bulk("W", a=3.18551, cubic=True)
    W_vac = W * 4
    del W_vac[64]

    dD = LML_W.get_dD(W_vac, linear=True)

    ML_calc = LML_W.make_lammpslib_calc()
    W_vac.calc = ML_calc
    forces = W_vac.get_forces()

    # most of the forces are almost zero, so relative tolerance does not work
    np.testing.assert_allclose(forces,
                               dD @ LML_W.linear_theta, atol=1.0e-3)

    dD_quad = LML_W.get_dD(W_vac, linear=False)

    # most of the forces are almost zero, so relative tolerance does not work
    np.testing.assert_allclose(forces,
                               dD_quad @ LML_W.theta_0_quad, atol=1.0e-3)