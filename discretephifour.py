import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal


class DiscretePhiFourSystem:
    """Object representing the topological discrete phi^four system."""

    def __init__(
        self, N: int, h: float, initial_value: float, vac: bool = False
    ) -> None:
        """Initialises the DiscretePhiFour class.

        Args:
            N: Number of spaces to calculate either side of the intial value.
            Total length of the system will be 2N+1.
            h: Lattice spacing of the system.
            initial_value: Value to start recursion from.
            Will be the central value.
            vac: Determines whether system is the vacuum solution phi=1.
        """
        self.N = N
        self.h = h
        self.initial_value = initial_value
        if vac == True:
            self.values = np.ones(2 * self.N + 1)
        else:
            self.values = np.array([self.initial_value])
            # Calculates the system using the formulae (19) and (20).
            for i in range(self.N):
                x = (
                    -(1 / 2)
                    * (
                        self.h * self.values[-1]
                        - np.sqrt(
                            -3 * self.h**2 * self.values[-1] ** 2
                            + 12 * self.h**2
                            + 36 * self.h * self.values[-1]
                            + 36
                        )
                        + 6
                    )
                    / self.h
                )
                self.values = np.append(
                    self.values, x
                )  # Appends phi_{n+1} to the end of the list.
                y = (
                    -(1 / 2)
                    * (
                        self.h * self.values[0]
                        + np.sqrt(
                            -3 * self.h**2 * self.values[0] ** 2
                            + 12 * self.h**2
                            - 36 * self.h * self.values[0]
                            + 36
                        )
                        - 6
                    )
                    / self.h
                )
                self.values = np.insert(
                    self.values, 0, y
                )  # Inserts phi_{n} at the start of the list.
        pass

    def D(self, n: int) -> float:
        """D is defined as the forward difference operator.

        Args:
            n: index
        """
        return (self.values[n + 1] - self.values[n]) / self.h

    def F(self, n: int) -> float:
        """F is defined as
        1 - (1/3)(phi_{n+1}^{2} + phi_{n+1}phi_{n} + phi_{n}^{2}).

        Args:
            n: index
        """
        return 1 - (1 / 3) * (
            self.values[n + 1] ** 2
            + self.values[n + 1] * self.values[n]
            + self.values[n] ** 2
        )

    def hessian(self) -> tuple[list[float], list[float]]:
        """Calculates the matrix of second partial derivatives of the
        potential energy functional.

        Returns:
            d: Elements on main diagonal of matrix.
            e: Elements on off-diagonal of matrix.
        """
        main_diag = []
        upper_and_lower_diag = []
        for i in range(2 * self.N + 1):
            if i == 0:
                x = (1 / self.h) + (self.h / 12) * (
                    self.values[1] ** 2
                    + 2 * self.values[1] * self.values[0]
                    + 2 * self.values[0] ** 2
                    - 2
                )
                y = (-1 / self.h) + (self.h / 12) * (
                    (self.values[i + 1] + self.values[i]) ** 2 - 1
                )
                main_diag.append(x)
                upper_and_lower_diag.append(y)
            elif 0 < i < 2 * self.N:
                x = (
                    (2 / self.h)
                    + (self.h / 3) * (self.values[i] ** 2 - 1)
                    + (self.h / 12)
                    * (
                        self.values[i - 1] ** 2
                        + self.values[i + 1] ** 2
                        + 2 * self.values[i] * (self.values[i + 1] + self.values[i - 1])
                    )
                )
                y = (-1 / self.h) + (self.h / 12) * (
                    (self.values[i + 1] + self.values[i]) ** 2 - 1
                )
                main_diag.append(x)
                upper_and_lower_diag.append(y)
            elif i == 2 * self.N:
                j = (1 / self.h) + (self.h / 12) * (
                    self.values[i - 1] ** 2
                    + 2 * self.values[i] * self.values[i - 1]
                    + 2 * self.values[i] ** 2
                    - 2
                )
                main_diag.append(j)
        return main_diag, upper_and_lower_diag

    def show(self):
        """Plots the kink."""
        plt.scatter(range(-self.N, self.N + 1), self.values)
        plt.xlabel("n")
        plt.ylabel(r"$\phi_{n}$")
        plt.show()

    def spectra(self) -> list:
        """Calculates the eigenvalues of the Hessian matrix.
        Uses Scipy's 'eigh_tridiagonal' function

        Returns:
            list: list of eigenvalues of the hessian.
        """
        return eigvalsh_tridiagonal(*self.hessian()

    def quantum_correction(self) -> float:
        """Calculates the quantum correction to the ground state energy.

        Returns:
            float: Quantum correction
        """
        return (1 / (2 * self.h)) * np.sum(np.sqrt(np.abs(self.spectra())))
