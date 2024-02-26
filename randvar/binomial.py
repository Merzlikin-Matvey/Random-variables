import numpy as np
import scipy as sp
from .random_variable import RandomVariable


class Binomial(RandomVariable):
    def __init__(self, n, p):
        self.n = n
        self.p = p
        super().__init__(np.arange(n + 1), [self.pmf(i) for i in range(n + 1)])

    def pmf(self, k):
        return sp.special.comb(self.n, k) * pow(self.p, k) * pow(1 - self.p, self.n - k)

    def mean(self):
        return self.n * self.p

    def variance(self):
        return self.n * self.p * (1 - self.p)

