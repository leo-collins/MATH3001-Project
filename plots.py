from discretephifour import DiscretePhiFourSystem
import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi=200)
N = 10

for h in [0.5, 1, 2]:
    kink = DiscretePhiFourSystem(10, h, 0)
    ax.scatter(range(-kink.N, kink.N + 1), kink.values, label=f"$h={h}$", s=20)
    ax.plot(range(-kink.N, kink.N + 1), kink.values)
ax.set_xlabel(r"$n$")
ax.set_ylabel(r"$\phi_{n}$")
ax.legend(fontsize="small")
plt.show()
