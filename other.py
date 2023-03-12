from discretephifour import DiscretePhiFourSystem


def potential_energy(kink: DiscretePhiFourSystem) -> float:
    sum = 0
    for n in range(2 * kink.N):
        sum += (1 / 2) * kink.D(n) ** 2 + (1 / 8) * kink.F(n) ** 2
    return kink.h * sum


def check_result(kink: DiscretePhiFourSystem):
    for n in range(2 * kink.N):
        print(kink.D(n) - (1 / 2) * kink.F(n))
