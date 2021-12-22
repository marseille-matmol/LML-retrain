import numpy as np
from scipy.linalg import orth


class LMLPotential:

    def __init__(self, filename):
        """
        Initialise the object by loading SNAP files in lammps format.
        Reads filename.snapparam and filename.snapcoeff.

        Parameters
        ----------
        filename : str path to load SNAP potential file in lammps format
        """
        # snapparam
        self.cutoff = None
        self.twojmax = None

        self.rfac0 = None
        self.rmin0 = None
        self.bzeroflag = None
        self.bnormflag = None


        self.symbol = None
        self.symbol_line = None
        self.linear_theta = None
        self.N_linear_desc = None

        self.load_lammps_snap_files(filename)
        # starting theta is linear coefficients extended with zeros
        self.theta_0_quad = np.zeros(self.N_linear_desc +
                                     (self.N_linear_desc * (self.N_linear_desc - 1)) // 2)
        self.theta_0_quad[:self.N_linear_desc] = self.linear_theta.copy()

        self.N_quad_desc = self.theta_0_quad.size
        self.N_linear_desc = self.linear_theta.size
        self.A_hard = None
        self.A_soft = None

        self.lambda_0 = 5.0e3
        self.lambda_S = 1.0

        self.target_forces = None
        self.target_dD = None

        self.new_theta = None

    def load_lammps_snap_files(self, filename):
        """
        Reads filename.snapparam and filename.snapcoeff

        Parameters
        ----------
        filename : str path to load SNAP potential file in lammps format.

        Returns
        -------
        None
        """
        # load parameters first
        with open(filename + ".snapparam") as param_file:
            for line in param_file.readlines():
                if "twojmax" in line:
                    self.twojmax = int(line.split(" ")[-1])
                if "rcutfac" in line:
                    self.cutoff = float(line.split(" ")[-1])
                if "rfac0" in line:
                    self.rfac0 = float(line.split(" ")[-1])
                if "rmin0" in line:
                    self.rmin0 = float(line.split(" ")[-1])
                if "bzeroflag" in line:
                    self.bzeroflag = int(line.split(" ")[-1])
                if "bnormflag" in line:
                    self.bnormflag = int(line.split(" ")[-1])

        with open(filename + ".snapcoeff") as coef_file:
            lines = coef_file.readlines()
            self.N_linear_desc = int(lines[4].split()[1])
            self.symbol_line = lines[5]
            self.symbol = self.symbol_line.split()[0]

        # load the coefficients
        self.linear_theta = np.loadtxt(filename + ".snapcoeff", skiprows=6)

        assert self.linear_theta.size == self.N_linear_desc

    def make_lammps_snap_files(self, filename, write_linear=False, folder=""):
        """
        Writes potential in lammps format files:
        filename.snapparam and filename.snapcoeff

        Parameters
        ----------
        filename : str
            Files for output filename.snapparam and filename.snapcoeff
        write_linear : bool
            If True writes initially loaded linear form of the potential,
            if False writes quadratic form of the retrained potential,
            default is False
        folder : str
            path to folder for the output.

        Returns
        -------

        """

        param_str = (f"#\n"
                     f"#\n"
                     f"# required\n"
                     f"rcutfac {self.cutoff}\n"
                     f"twojmax {self.twojmax}\n"
                     f"\n"
                     f"# optional\n")
        if self.rfac0 is not None:
            param_str += f"rfac0 {self.rfac0}\n"
        if self.rmin0 is not None:
            param_str += f"rmin0 {self.rmin0}\n"
        if self.bzeroflag is not None:
            param_str += f"bzeroflag {self.bzeroflag}\n"

        param_str += f"quadraticflag {int(not write_linear)}\n"

        if self.bnormflag is not None:
            param_str += f"bnormflag {self.bnormflag}\n"

        f = open(folder + f'{filename}.snapparam', 'w')
        f.write(param_str)
        f.close()

        if write_linear:
            header_str = (f"# autogen test\n"
                          f"# \n"
                          f"# LAMMPS SNAP coefficients for {self.symbol}\n"
                          f"\n"
                          f"1 {self.N_linear_desc}\n"
                          f"{self.symbol_line.strip()}")

            np.savetxt(folder + f"{filename}.snapcoeff", self.linear_theta,
                       header=header_str, comments="")
        else:
            header_str = (f"# autogen test\n"
                          f"# \n"
                          f"# LAMMPS SNAP coefficients for {self.symbol}\n"
                          f"\n"
                          f"1 {self.N_quad_desc}\n"
                          f"{self.symbol_line.strip()}")
            if self.new_theta is not None:
                np.savetxt(folder + f"{filename}.snapcoeff", self.new_theta,
                           header=header_str, comments="")
            else:
                np.savetxt(folder + f"{filename}.snapcoeff", self.theta_0_quad,
                           header=header_str, comments="")

    def make_lammpslib_calc(self, use_default_linear=False):
        """
        Created LAMMPSlib calculator that can be used with ase.atoms

        Parameters
        ----------
        use_default_linear : bool
            If True uses initially loaded linear form of the potential,
            if False uses quadratic form of the retrained potential,
            default is False.

        Returns
        -------
        ase.calculator

        """
        import os
        from ase.calculators.lammpslib import LAMMPSlib

        filename = "LML"
        self.make_lammps_snap_files(filename=filename,
                                    write_linear=use_default_linear)

        pot_files = [f"{filename}.snapcoeff", f"{filename}.snapparam"]
        # pot_paths = [os.path.join(pot_folder, pot_file for pot_file in pot_files]
        pot_path = " ".join(pot_files)
        # print(pot_path)
        lmpcmds = ["pair_style snap",
                   f"pair_coeff * * {pot_path} {self.symbol}"]

        ML_calc = LAMMPSlib(lmpcmds=lmpcmds,
                            atom_types={self.symbol: 1}, keep_alive=True)

        from ase.build import bulk
        bulk_atoms = bulk(self.symbol)
        bulk_atoms.calc = ML_calc
        bulk_atoms.get_potential_energy()
        for pot_file in pot_files:
            os.remove(pot_file)

        return ML_calc

    def get_D(self, atoms, linear=False):
        """
        Calculates bispectrum (SNAP) descriptor vector from ase.atoms
        Parameters
        ----------
        atoms : ase.atoms object
            atoms object containing atomic positions
        linear: bool
            If True uses linear form of the descriptor
            (controls the size of the vector)

        Returns
        -------
        D : np.array
            Array of descriptor values with shape (N_atoms, N_desc)

        """
        LML_calc = self.make_lammpslib_calc(use_default_linear=linear)
        atoms.calc = LML_calc
        E = atoms.get_potential_energy()

        LML_calc.lmp.commands_string(f"compute D all sna/atom {self.cutoff} "
                                     f"{self.rfac0} {self.twojmax} 0.5 1 "
                                     f"quadraticflag {int(not linear)}")
        # LML_calc.lmp.commands_string("compute E all pe/atom")
        LML_calc.lmp.commands_string("run 0")
        # energies = np.ctypeslib.as_array(LML_calc.lmp.gather('c_E', 1, 1))
        if linear:
            D = np.ctypeslib.as_array(LML_calc.lmp.gather('c_D', 1,
                                                          self.N_linear_desc - 1))
            D = D.reshape((-1, self.N_linear_desc - 1))
        else:
            D = np.ctypeslib.as_array(LML_calc.lmp.gather('c_D',
                                                          1, self.N_quad_desc - 1))
            D = D.reshape((-1, self.N_quad_desc - 1))

        # in our notation we have to ad extra ones
        # for the first (zero) descriptor dimension
        D = np.hstack((np.ones(shape=(D.shape[0], 1)), D))

        LML_calc.lmp.close()
        return D

    def get_dD(self, atoms, linear=False):
        """
        Computes descriptor derivatives for a given ase.atoms object
        Parameters
        ----------
        atoms : ase.atoms object
            atoms object containing atomic positions.
        linear: bool
            If True uses linear form of the descriptor
            (controls the size of the vector).

        Returns
        -------
        dD : np.array
            Array of descriptor derivatives values
            with shape (N_atoms, 3, N_desc).
        """
        LML_calc = self.make_lammpslib_calc(use_default_linear=linear)
        atoms.calc = LML_calc
        E = atoms.get_potential_energy()

        LML_calc.lmp.commands_string(f"compute dD all snad/atom {self.cutoff} "
                                     f"{self.rfac0} {self.twojmax} 0.5 1 "
                                     f"quadraticflag {int(not linear)}")
        LML_calc.lmp.commands_string("run 0")

        if linear:
            dD = np.ctypeslib.as_array(LML_calc.lmp.gather('c_dD', 1,
                                                          3 * (self.N_linear_desc - 1)))
            dD = dD.reshape((-1, self.N_linear_desc - 1))
        else:
            dD = np.ctypeslib.as_array(LML_calc.lmp.gather('c_dD',
                                                          1,
                                                          3 * (self.N_quad_desc - 1)))
            dD = dD.reshape((-1, self.N_quad_desc - 1))

        # in our notation we have to ad extra zeros
        # for the first (zero) descriptor dimension
        dD = np.hstack((np.zeros(shape=(dD.shape[0], 1)), dD))
        dD = dD.reshape((len(atoms), 3, dD.shape[1]))
        LML_calc.lmp.close()
        return dD

    def write_retrained_potential(self, title="dislocation", folder=""):
        """
        Write retrained potential in lammps format.
        Parameters
        ----------
        title : str
            title of the potential to add in the final filename.
        folder : str
            path to the folder.

        Returns
        -------
        None
        """
        if self.new_theta is None:
            raise RuntimeError("New potential is not created")

        pot_name = "qSNAP" + title + \
                   f"_Lam0_{self.lambda_0:g}" + \
                   f"_LamS_{self.lambda_S:g}".replace("0.", "0")

        self.make_lammps_snap_files(filename=pot_name, folder=folder)

    def plot_w_coefs(self, ax=None):
        """
        plots coefficient of the original and retrained potentials.
        Parameters
        ----------
        ax : matplotlib.axes
            plot on existing axes, otherwise create a new figure

        Returns
        -------
        None
        """

        import matplotlib.pyplot as plt

        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        ax.semilogx(self.theta_0_quad[1:],
                    '--', label="Original LML values $\Theta_0$", lw=1.75,
                    zorder=10)

        if self.new_theta is not None:
            ax.semilogx(self.new_theta[1:],
                        "-", label="Retrained LML values $\Theta_0+\delta\Theta$", lw=2)

        ax.set_ylabel('Magnitude')
        ax.set_xlabel(f"Bispectrum component; N={self.N_quad_desc}")

        ax.legend()
        ax.set_xscale("symlog", linthresh=55)
        if fig is not None:
            fig.show()

    def get_projection(self, matrix_A, size=None):
        """
        Build projection of a constraint matrix using scipy.linalg.orth.
        Is used for getting projection matrix for hard constraints.

        Parameters
        ----------
        matrix_A : np.array
            Matrix to build projection.
        size : int
            Size of the matrix.
            Default is None and the size is defined by self.N_quad_desc

        Returns
        -------
        P : np.array
            Projection matrix.
        """

        if size is None:
            size = self.N_quad_desc

        if matrix_A is not None:
            if matrix_A.shape[0] == size:
                e = orth(matrix_A)
            else:
                e = orth(matrix_A.T)

            P = e @ e.T
        else:
            P = np.zeros((size, size))

        return P

    def refit_pot_from_forces(self, target_forces, target_dD,
                              lambda_0=None, lambda_S=None,
                              use_Ps=False):
        """
        Function to perform constrained retraining, i.e.
        find a solution for minimizing loss function given
        in equation (3) of the manuscript: https://arxiv.org/abs/2111.11262.

        Parameters
        ----------
        target_forces : np.array
            array of target forces with shape (N_atoms, 3),
            For the manuscript we used QM/ML forces.
        target_dD : np.array
            target descriptor derivatives with shape (N_atoms, 3, N_desc).
        lambda_0 : flaot
            Regularisation coefficient, see eq 3.
        lambda_S : float
            Soft constraints coefficient, see eq 3.
        use_Ps : bool
            If True soft constrained matrix is projected as well.
             This is a testing option and in the retraining
             reported in the paper it was not used.

        Returns
        -------
        None
            But updates the self.new_theta.

        """
        np.testing.assert_array_equal(target_forces.shape,
                                      (target_dD @ self.theta_0_quad).shape)

        self.target_forces = target_forces
        self.target_dD = target_dD

        if self.A_soft is None:
            raise RuntimeError("Soft constraints are empty!")
        if self.A_hard is None:
            raise RuntimeError("Hard constraints are empty!")

        try:
            assert (self.N_quad_desc == self.A_hard.shape[1])
            assert (self.N_quad_desc == self.A_soft.shape[1])

        except AssertionError:

            raise RuntimeError("Shape of constraints does not correspond "
                               "to the size of descriptors:\n"
                               f"N_desc: {self.N_quad_desc}\n"
                               f"A_hard: {self.A_hard.shape[1]}\n"
                               f"A_soft: {self.A_soft.shape[1]})")

        if lambda_0 is not None:
            self.lambda_0 = lambda_0
        if lambda_S is not None:
            self.lambda_S = lambda_S

        # Hard projection
        P_H = self.get_projection(self.A_hard)
        Q_H = np.eye(self.N_quad_desc) - P_H

        if use_Ps:
            P_S = self.get_projection(self.A_soft, self.N_quad_desc)
            D_soft = int(np.diagonal(P_S).sum())
            print(
                f"Soft constraints dimensionality: {D_soft:d}/{self.N_quad_desc}")


        D_hard = int(Q_H.shape[0] - np.diagonal(Q_H).sum())
        print(f"Hard constraints dimensionality: {D_hard:d}/{self.N_quad_desc}")

        if use_Ps:
            M_S = Q_H @ self.A_soft.T @ self.A_soft @ Q_H

            # Replace this with P_S
            ev = np.linalg.eigvalsh(M_S)
            ev = ev[ev > 1.0].mean()
            M_S = Q_H @ P_S @ Q_H * ev

        else:
            M_S = self.A_soft.T @ self.A_soft
            print(
                f"Soft constraints "
                f"dimensionality: {np.linalg.matrix_rank(M_S):d}"
                f"/{self.N_quad_desc}")
        # {\bf b} = \delta{\bf F}_{\rm QMML}\cdot\nabla{\bf D}{\bf Q}_{\rm H}
        # lammps compute snad/atom command computes -dD (negative derivatives)
        # thus dD * Theta is forces and the sign is changed compared to
        # eq (3) of the paper

        # flatten the 3D forces and descriptors
        target_dD = target_dD.reshape((target_forces.flatten().shape[0], -1))
        assert target_dD.shape[1] == self.N_quad_desc

        b = Q_H @ target_dD.T @ (
                    target_forces.flatten() - target_dD @ self.theta_0_quad)

        # {\bf M} = {\bf Q}_{\rm H}[\nabla{\bf D}]^\top\nabla{\bf D}{\bf Q}_{\rm H}
        M = Q_H @ target_dD.T @ target_dD @ Q_H

        if use_Ps:
            # {\bf M} = \lambda_0\mathbb{I}
            M += self.lambda_0 * np.eye(Q_H.shape[0]) + self.lambda_S * M_S

            # {\bf M} += \lambda_S{\bf Q}_{\rm H}{\bf P}_S{\bf Q}_{\rm H}
            M += self.lambda_S * M_S  # A_soft.T@A_soft

        else:
            M += self.lambda_S * Q_H @ M_S @ Q_H + \
                 self.lambda_0 * np.eye(Q_H.shape[0])
            # mod = linear_model.BayesianRidge()#lambda_init=lambda_0)
        # mod.fit(M,b)
        dTheta = np.linalg.solve(M, b)

        # print(dTheta @ Q_H @ self.A_soft.T @ self.A_soft @ Q_H @ dTheta,
        #      self.lambda_S * dTheta @ Q_H @ P_S @ Q_H @ dTheta)

        self.new_theta = self.theta_0_quad.copy() + Q_H @ dTheta

        return None

    def plot_target_force_error(self, target_positions, components=True,
                                ax=None):
        """
        Function to plot force error on target configurations

        Parameters
        ----------
        target_positions : np.array
            Target positions with shape (N_atoms, 3)
        components : bool
            Controls the way error is computer.
            If True error on force components is computed,
            otherwise error on total force is computed.
            Both ways give very similar results.
        ax : matplotlib.axes
            plot on existing axes, otherwise create a new figure.

        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt

        if self.new_theta is None:
            raise RuntimeError("The potential was not retrained!")
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        # x-y distance from the core (cell center)
        r = np.linalg.norm(target_positions[:, :2] -
                           target_positions[:, :2].mean(axis=0), axis=1)

        old_forces = (self.target_dD @ self.theta_0_quad).reshape((-1, 3))
        new_forces = (self.target_dD @ self.new_theta).reshape((-1, 3))

        if components:
            delta_new_forces = np.abs(new_forces - self.target_forces)
            delta_old_forces = np.abs(old_forces - self.target_forces)

            ax.plot(np.repeat(r, 3), delta_old_forces.flatten(), "o",
                    label=f"fmax {delta_old_forces.flatten().max():2.2g} " +
                          r"eV/$\rm\AA$")
            ax.plot(np.repeat(r, 3), delta_new_forces.flatten(), ".",
                    label=f"fmax {delta_new_forces.flatten().max():2.2g} " +
                          r"eV/$\rm\AA$")

        else:
            delta_new_forces = np.linalg.norm(new_forces - self.target_forces,
                                              axis=1)
            delta_old_forces = np.linalg.norm(old_forces - self.target_forces,
                                              axis=1)

            ax.plot(r, delta_old_forces, "o",
                    label=f"fmax {delta_old_forces.max():2.2g} " +
                          r"eV/$\rm\AA$")
            ax.plot(r, delta_new_forces, ".",
                    label=f"fmax {delta_new_forces.max():2.2g} " +
                          r"eV/$\rm\AA$")

        ax.set_title("Error to DFT forces from QM/ML")

        ax.set_ylabel(r"Force error eV/$\rm\AA$")
        ax.set_xlabel(r"Distance from dislocation $\rm\AA$")
        ax.legend()
        if fig is not None:
            fig.show()

    def plot_MD_test_train(self, dD_test, dD_train=None,
                           ax=None, force_error=0.1, title=""):
        """
        Plots test train force error given arrays of descriptor derivatives.

        Parameters
        ----------
        dD_test : np.array
            Descriptor derivatives on test configuration.
        dD_train : np.array
            Descriptor derivatives on train configuration.
        ax : matplotlib.axes
            plot on existing axes, otherwise create a new figure.
        force_error : float
            Value to plot as an error interval.
        title : str
            Title of the figure.

        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt

        if self.new_theta is None:
            raise RuntimeError("The potential was not retrained!")

        if ax is None:
            fig, ax = plt.subplots()

        if dD_train is None:
            dD_train = self.A_soft

        old_train_forces = dD_train @ self.theta_0_quad
        new_train_forces = dD_train @ self.new_theta

        train_rms = (old_train_forces - new_train_forces).std()

        ax.scatter(old_train_forces, new_train_forces,
                   label=f'Train, RMSE = {train_rms:2.2g}' + r' eV/$\rm\AA$')

        old_test_forces = dD_test @ self.theta_0_quad
        new_test_forces = dD_test @ self.new_theta

        test_rms = (old_test_forces - new_test_forces).std()

        ax.scatter(old_test_forces, new_test_forces,
                   label=f'Test, RMSE = {test_rms:2.2g}' + r' geV/$\rm\AA$')

        error_points = np.linspace(old_train_forces.min(),
                                   old_train_forces.max(), 11)

        ax.fill_between(error_points,
                        error_points - force_error,
                        error_points + force_error,
                        facecolor='k', alpha=0.2,
                        label=r'%2.2geV/$\rm\AA$ error' % force_error)

        ax.set_title(title)
        ax.legend(fontsize=10)
        ax.set_xlabel(r"Original LML Forces (eV/$\rm\AA$)", fontsize=10)
        ax.set_ylabel(r"Retrained QML Forces (eV/$\rm\AA$)", fontsize=10)


    def plot_eos_data(self, configurations, ax=None,
                      title=""):
        """
        Evaluates and plot total energy values with original
        and retrained potentials on atoms object
        of Equation of State configurations:

        Parameters
        ----------
        configurations : list of ase.atoms
            List of configurations with strain
            labels in ase.atoms.info["strain_info"].
        ax : matplotlib.axes
            plot on existing axes, otherwise create a new figure.
        title : str
            Title of the figure

        Returns
        -------
        None
        """
        if self.new_theta is None:
            raise RuntimeError("The potential was not retrained!")

        import matplotlib.pyplot as plt

        old_calc = self.make_lammpslib_calc(use_default_linear=True)
        new_calc = self.make_lammpslib_calc(use_default_linear=False)

        labels = np.array(
            [config.info["strain_type"] for config in configurations])

        if ax is None:
            fig, ax = plt.subplots()

        indexes = np.unique(labels, return_index=True)[1]
        # this is to keep the order of the labels
        labels = [labels[index] for index in sorted(indexes)]
        handles = []
        for strain_type in labels:

            strains = []
            old_energies = []
            new_energies = []
            configs = [config for config in configurations if
                       config.info["strain_type"] == strain_type]

            for config in configs:
                strains.append(config.info["strain_value"])
                config.calc = old_calc
                old_energies.append(config.get_potential_energy())

                config = config.copy()
                config.calc = new_calc
                new_energies.append(config.get_potential_energy())

            old_energies = np.array(old_energies) * 1.0e3  # show meV
            old_energies -= old_energies.min()

            new_energies = np.array(new_energies) * 1.0e3  # show meV
            new_energies -= new_energies.min()
            strains = np.array(strains) * 1.0e2  # show %

            ax.scatter(strains, new_energies / len(configs[0]), s=50)
            ax.plot(strains, old_energies / len(configs[0]),
                    lw=2.0, label=strain_type.replace("bar", r"\bar"))

            handles.append(ax.plot([], [], "-o", c=ax.get_lines()[-1].get_color(),
                                   label=strain_type.replace("bar", r"\bar"))[0])

        ax.set_xlabel(r'Strain $\epsilon$ (%)')
        ax.set_xticks(strains[::4])

        ax.legend(handles=handles,
                  title="Equation of state\nlines: original LML\ndots: retrained LML")

        ax.set_ylabel(r"$\Delta E_{\rm coh}$ (meV/atom)")

        ax.set_title(title)
        ax.set_ylim(-0.1, 2.8)
        new_calc.lmp.close()
        old_calc.lmp.close()

    def plot_MD_test_train_atoms(self, test_config, train_configs, ax=None,
                                 title=""):
        """
        Plots test train force error given ase.atoms objects.
        Parameters
        ----------
        test_config : ase.atoms
            test configuration.
        train_configs : list of ase.atoms
            training configurations.
        ax : matplotlib.axes
            plot on existing axes, otherwise create a new figure.
        title : str
            Title of the figure

        Returns
        -------
        None
        """

        if self.new_theta is None:
            raise RuntimeError("The potential was not retrained!")

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        old_calc = self.make_lammpslib_calc(use_default_linear=True)
        new_calc = self.make_lammpslib_calc(use_default_linear=False)

        old_train_forces = []
        new_train_forces = []

        for config in train_configs:
            config.calc = old_calc
            old_train_forces.append(config.get_forces().flatten())

            config = config.copy()
            config.calc = new_calc
            new_train_forces.append(config.get_forces().flatten())

        new_train_forces = np.concatenate(new_train_forces)
        old_train_forces = np.concatenate(old_train_forces)

        train_rms = (old_train_forces - new_train_forces).std()
        print(f"Train RMSE = {train_rms:2.2g} eV/A")
        ax.scatter(old_train_forces, new_train_forces,
                   label=f'Train, RMSE = {train_rms:2.2g}' + r' eV/$\rm\AA$')

        test_config.calc = old_calc
        old_test_forces = test_config.get_forces().flatten()

        test_config.calc = new_calc
        new_test_forces = test_config.get_forces().flatten()

        test_rms = (old_test_forces - new_test_forces).std()
        print(f'Test RMSE = {test_rms:2.2g} eV/A')
        ax.scatter(old_test_forces, new_test_forces,
                   label=f'Test, RMSE = {test_rms:2.2g}' + r' eV/$\rm\AA$')

        error_points = np.linspace(old_train_forces.min(),
                                   old_train_forces.max(), 11)
        """
        ax.fill_between(error_points,
                        error_points - force_error,
                        error_points + force_error,
                        facecolor='k', alpha=0.2,
                        label=r'%2.2g eV/$\rm\AA$ error' % force_error)
        """
        #ax.set_title(title)
        # ymin, ymax = ax.get_ylim()
        # ax.set_ylim((ymin, 1.2 * ymax)
        ax.legend( title=title)

        ax.set_xlabel(r"Original LML Forces (eV/$\rm\AA$)")
        ax.set_ylabel(r"Retrained LML Forces (eV/$\rm\AA$)")

        new_calc.lmp.close()
        old_calc.lmp.close()

    def plot_NEB_data(self, images, ax=None):
        """
        Evaluates and plots NEB path for original and retrained potentials.

        Parameters
        ----------
        images : list of ase.atoms
            NEB path images.
        ax : matplotlib.axes
            plot on existing axes, otherwise create a new figure.

        Returns
        -------
        None
        """
        if self.new_theta is None:
            raise RuntimeError("The potential was not retrained!")

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        old_calc = self.make_lammpslib_calc(use_default_linear=True)
        new_calc = self.make_lammpslib_calc(use_default_linear=False)

        old_E = []
        new_E = []

        for image in images:
            image.calc = old_calc
            old_E.append(image.get_potential_energy())

            image.calc = new_calc
            new_E.append(image.get_potential_energy())

        new_E = np.array(new_E)
        new_E -= new_E[0]

        old_E = np.array(old_E)
        old_E -= old_E[0]

        ax.plot(np.arange(len(images)), old_E, 'o-', lw=2,
                     label=r"Original $\Theta_0$ Potential")
        ax.plot(np.arange(len(images)), new_E, 's--', lw=2,
                     label=r"Refitted $\Theta_0+\delta\Theta$ Potential")

        ax.legend(fontsize=10)

        ax.set_xlabel(r"NEB image", fontsize=10)
        ax.set_ylabel(r"Energy Difference [eV]", fontsize=10)

        new_calc.lmp.close()
        old_calc.lmp.close()