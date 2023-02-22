import matplotlib.pyplot as plt
import numpy as np

h = 1  # lattice spacing
n = 10
initial_value = 0


def recur_forward(n):
    values = np.array([initial_value])
    for i in range(1, n + 1):
        np.append(
            values,
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
            / h,
        )
    return values


def recur_backward(n):
    values = np.array([initial_value])
    for i in range(1, n + 1):
        np.append(
            values,
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
            / h,
        )
    return values


# hZ = [h*n for n in ]
values = np.array(recur_backward(n)[:0:-1] + recur_forward(n))
plt.scatter(range(2 * n + 1), values)
plt.show()
