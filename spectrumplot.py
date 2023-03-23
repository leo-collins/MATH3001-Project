from discretephifour import DiscretePhiFourSystem
import matplotlib.pyplot as plt
import numpy as np

h = 2
# N = 100

# vac_sol = DiscretePhiFourSystem(N, h, 0, vac=True)
# plt.scatter(range(2*N + 1), vac_sol.spectra())
# plt.show()

# kink = DiscretePhiFourSystem(N, h, 0.2)
# plt.scatter(range(2*N + 1), kink.spectra())
# plt.show()

spectrum = []
fig, ax = plt.subplots(dpi=400)

for i in range(40):
    kink = DiscretePhiFourSystem(i + 1, h, 0, vac=False)
    ax.plot([2 * i + 3] * (2 * i + 3), kink.spectra(), "o", markersize=1, color="black")
ax.set_xlabel("N")
ax.set_ylabel("spectra")
plt.show()
