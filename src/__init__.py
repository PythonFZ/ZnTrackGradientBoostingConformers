import functools

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zntrack
from ase.calculators.singlepoint import SinglePointCalculator
from dscribe.descriptors import SOAP
from rdkit import Chem
from rdkit2ase import rdkit2ase
from rdkit.Chem import AllChem
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xtb.ase.calculator import XTB


@functools.singledispatch
def freeze_copy_atoms(atoms) -> ase.Atoms:
    # TODO can we add the name of the original calculator?
    result = atoms.copy()
    result.calc = SinglePointCalculator(result, **atoms.calc.results)
    return result


@freeze_copy_atoms.register
def _(atoms: list) -> list[ase.Atoms]:
    return [freeze_copy_atoms(x) for x in atoms]


class CreateConformers(zntrack.Node):
    smiles: str = zntrack.params("CCO")
    num_confs: int = zntrack.params(200)
    random_seed: int = zntrack.params(42)
    max_attempts: int = zntrack.params(100)

    conformers_path: str = zntrack.outs_path(zntrack.nwd / "conformers.xyz")

    def run(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)

        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=self.num_confs,
            randomSeed=self.random_seed,
            maxAttempts=self.max_attempts,
        )
        atoms = [rdkit2ase(x) for x in mol.GetConformers()]

        ase.io.write(self.conformers_path, atoms)

    @property
    def atoms(self) -> list[ase.Atoms]:
        return list(ase.io.iread(self.conformers_path))


class ComputeEnergies(zntrack.Node):
    method: str = zntrack.params("GFN1-xTB")
    data: list[ase.Atoms] = zntrack.deps()

    atoms: list[ase.Atoms] = zntrack.outs()
    energies: pd.DataFrame = zntrack.plots(
        x_label="# Conformation", y_label="Energy / eV"
    )

    def run(self) -> None:
        for atoms in self.data:
            atoms.calc = XTB(method=self.method)
            atoms.get_potential_energy()
        self.atoms = freeze_copy_atoms(self.data)
        self.energies = pd.DataFrame([x.get_potential_energy() for x in self.atoms])


class CreateDatasets(zntrack.Node):
    atoms: list[ase.Atoms] = zntrack.deps()

    r_cut: float = zntrack.params(5.0)
    n_max: int = zntrack.params(5)
    l_max: int = zntrack.params(5)
    test_size: float = zntrack.params(0.2)
    random_state: int = zntrack.params(42)

    X_train: np.ndarray = zntrack.outs()
    X_test: np.ndarray = zntrack.outs()
    y_train: np.ndarray = zntrack.outs()
    y_test: np.ndarray = zntrack.outs()

    def run(self) -> None:
        soap = SOAP(
            species=set(self.atoms[0].get_chemical_symbols()),
            periodic=False,
            r_cut=self.r_cut,
            n_max=self.n_max,
            l_max=self.l_max,
        )
        features = soap.create(self.atoms)
        features = features.reshape((len(self.atoms), -1))
        targets = np.array([atoms.get_potential_energy() for atoms in self.atoms])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, targets, test_size=self.test_size, random_state=self.random_state
        )


class TrainModel(zntrack.Node):
    X_train: np.ndarray = zntrack.deps()
    y_train: np.ndarray = zntrack.deps()

    n_estimators: int = zntrack.params(200)
    learning_rate: float = zntrack.params(0.1)
    random_state: int = zntrack.params(42)

    model = zntrack.outs()

    def run(self) -> None:
        # Gradient Boosting Regression
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self.model.fit(self.X_train, self.y_train)


class EvaluateModel(zntrack.Node):
    model = zntrack.deps()
    X_test: np.ndarray = zntrack.deps()
    y_test: np.ndarray = zntrack.deps()

    mse: dict = zntrack.metrics()
    correlation_plot_path: str = zntrack.plots_path(zntrack.nwd / "correlation.png")

    def run(self) -> None:
        predictions = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)

        self.mse = {"MSE": mse}

        fig, ax = plt.subplots()
        ax.scatter(self.y_test, predictions, marker="x", color="red")
        ax.plot(self.y_test, self.y_test, color="black")
        ax.set_xlabel("xTB Energy")
        ax.set_ylabel("Predicted Energy")
        fig.savefig(self.correlation_plot_path, bbox_inches="tight")
