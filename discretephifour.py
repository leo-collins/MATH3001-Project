import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh_tridiagonal


class DiscretePhiFour:
    """Object representing the topological discrete phi^four system."""

    def __init__(self, N: int, h: float, initial_value: float) -> None:
        """Initialises the DiscretePhiFour class.

        Args:
            N: Number of spaces to calculate either side of the intial value.
            Total length of the system will be 2N+1.
            h: Lattice spacing of the system.
            initial_value: Value to start recursion from.
            Will be the central value.
        """
        self.N = N
        self.h = h
        self.initial_value = initial_value
        self.values = np.array([self.initial_value])
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
            self.values = np.append(self.values, x)
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
            self.values = np.insert(self.values, 0, y)
        pass

    def D(self, n: int) -> float:
        """D is defined as the forward difference operator.

        Args:
            n: index

        Returns:
            float: (phi_{n+1} - phi_{n}) / h
        """
        return (self.values[n + 1] - self.values[n]) / self.h

    def F(self, n: int) -> float:
        """

        Args:
            n: index

        Returns:
            float: _description_
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
                x = 2 * ((1 / self.h) + (self.h / 6) * (self.values[i] ** 2 - 1)) + (
                    self.h / 12
                ) * (
                    self.values[i - 1] ** 2
                    + self.values[i + 1] ** 2
                    + 2 * self.values[i] * (self.values[i + 1] + self.values[i - 1])
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
        plt.scatter(range(2 * self.N + 1), self.values)
        plt.show()

    def hessian_eigenvalues(self):
        return eigh_tridiagonal(*self.hessian())[0]

    def quantum_correction(self):
        return (1 / (2 * self.h)) * np.sum(np.sqrt(np.abs(self.hessian_eigenvalues())))


# def potential_energy(discrete_system: np.ndarray) -> float:
#     sum = 0
#     for n in range(len(discrete_system) - 1):
#         sum += (1 / 2) * D(discrete_system, n) ** 2 + (1 / 8) * F(
#             discrete_system, n
#         ) ** 2
#     return h * sum


# def check_result(discrete_system: npt.NDArray):
#     for n in range(len(discrete_system) - 1):
#         print(D(discrete_system, n) - (1 / 2) * F(discrete_system, n))

fig, ax = plt.subplots(dpi=200)
N = 10

for h in np.linspace(1.2, 1.7, num=6):
    values = []
    b_h = []
    kink_vacuum = DiscretePhiFour(N, h, 0)
    kink_vac_correction = kink_vacuum.quantum_correction()
    for b in np.linspace(
        -kink_vacuum.values[kink_vacuum.N + 1],
        kink_vacuum.values[kink_vacuum.N + 1],
        num=100,
    ):
        kink_translated = DiscretePhiFour(N, h, b)
        # values.append(
        #     (
        #         np.sum(np.sqrt(np.abs(kink_translated.hessian_eigenvalues())))
        #         - np.sqrt(np.sum(np.abs(kink_vacuum.hessian_eigenvalues())))
        #     )
        #     - kink_vac_correction
        # )
        values.append(kink_translated.quantum_correction() - kink_vac_correction)
        b_h.append(b / kink_vacuum.values[kink_vacuum.N + 1])
    ax.plot(b_h, values, label=f"h={h:.1f}")

ax.set_title(f"n={N}")
ax.set_xlabel("b/h")
ax.set_ylabel("e(b) - e(0)")
ax.legend(fontsize="x-small")
plt.show()
