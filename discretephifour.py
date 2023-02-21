import numpy as np
import matplotlib.pyplot as plt

h = 0.1
n = 100


def recur2(n):
    values = [0.5]
    for i in range(1, n + 1):
        values.append(
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
    return values


def recur3(n):
    values = [0.5]
    for i in range(1, n + 1):
        values.append(
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
    return values


values = recur3(n)[:0:-1] + recur2(n)
plt.scatter(range(2 * n + 1), values)
plt.show()
