import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh_tridiagonal

h = 1  # lattice spacing
n = 10
initial_value = 0.5


def recur_forward(n: int) -> np.ndarray:
    values = np.array([initial_value])
    for i in range(1, n + 1):
        j = (
            -(1 / 2)
            * (
                h * values[-1]
                - np.sqrt(
                    -3 * h**2 * values[-1] ** 2
                    + 12 * h**2
                    + 36 * h * values[-1]
                    + 36
                )
                + 6
            )
            / h
        )
        values = np.append(values, j)
    return values


def recur_backward(n: int) -> np.ndarray:
    values = np.array([initial_value])
    for i in range(1, n + 1):
        j = (
            -(1 / 2)
            * (
                h * values[-1]
                + np.sqrt(
                    -3 * h**2 * values[-1] ** 2
                    + 12 * h**2
                    - 36 * h * values[-1]
                    + 36
                )
                - 6
            )
            / h
        )
        values = np.append(values, j)
    return values


def potential_energy(discrete_system: np.ndarray) -> float:
    sum = 0
    for n in range(len(discrete_system) - 1):
        sum += (
            (1 / h**2) * (discrete_system[n + 1] - discrete_system[n])
        ) ** 2 + 0.25 * (
            1
            - (1 / 3)
            * (
                discrete_system[n + 1] ** 2
                + discrete_system[n + 1] * discrete_system[n]
                + discrete_system[n] ** 2
            )
        ) ** 2
    return 0.5 * h * sum


def f(x):
    return h**2 * 6 * x**2


# hZ = [h*n for n in ]
values = np.append(recur_backward(n)[:0:-1], recur_forward(n - 1))
W_mat = np.diag(f(values)) - np.eye(2 * n, k=1) - np.eye(2 * n, k=-1)

w, v = eigh_tridiagonal(f(values), np.ones(19) * -1)
np.allclose(W_mat @ v - v @ np.diag(w), np.zeros((20, 20)))

plt.scatter(range(2 * n), values)
plt.show()
