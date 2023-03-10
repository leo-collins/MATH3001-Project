from discretephifour import DiscretePhiFourSystem
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(dpi=200)
N = 10

for h in np.linspace(1.2, 1.7, num=6):
    values = []
    b_h = []
    kink_vacuum = DiscretePhiFourSystem(N, h, 0)
    kink_vac_correction = kink_vacuum.quantum_correction()
    for b in np.linspace(
        -kink_vacuum.values[kink_vacuum.N + 1],
        kink_vacuum.values[kink_vacuum.N + 1],
        num=100,
    ):
        kink_translated = DiscretePhiFourSystem(N, h, b)
        values.append(kink_translated.quantum_correction() - kink_vac_correction)
        b_h.append(b / kink_vacuum.values[kink_vacuum.N + 1])
    ax.plot(b_h, values, label=f"h={h:.1f}")

ax.set_title(f"n={N}")
ax.set_xlabel("b/h")
ax.set_ylabel("e(b) - e(0)")
ax.legend(fontsize="x-small")
plt.show()
