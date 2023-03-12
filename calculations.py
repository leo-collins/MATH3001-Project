from discretephifour import DiscretePhiFourSystem
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(dpi=200)
N = 100


def E(b: float, vac_eigs: list):
    kink_b = DiscretePhiFourSystem(N, h, b)
    return np.sum(np.sqrt(np.abs(kink_b.spectra())) - np.sqrt(vac_eigs))


for h in np.linspace(1.2, 1.7, num=6):
    values = []
    b_h = []
    vac_eigs = DiscretePhiFourSystem(N, h, 0, vac=True).spectra()
    kink_0 = DiscretePhiFourSystem(N, h, 0)
    for b in np.linspace(kink_0.values[N - 1], kink_0.values[N + 1], num=100):
        values.append(E(b, vac_eigs) - E(0, vac_eigs))
        b_h.append(b / kink_0.values[kink_0.N + 1])
    ax.plot(b_h, values, label=f"h={h:.1f}")

ax.set_title(f"N={N}")
ax.set_xlabel("b/h")
ax.set_ylabel("e(b) - e(0)")
ax.legend(fontsize="x-small")
plt.show()
