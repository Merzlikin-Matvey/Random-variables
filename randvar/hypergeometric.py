import numpy as np
import scipy as sp
from .random_variable import RandomVariable


class Hypergeometric(RandomVariable):
    def __init__(self, N, K, n):
        self.N = N
        self.K = K
        self.n = n
        self.values = np.arange(max(0, n - (N - K)), min(K, n) + 1)
        self.probabilities = np.array([self.pmf(x) for x in self.values])
        super().__init__(self.values, self.probabilities)

    def pmf(self, x):
        return (sp.special.comb(self.K, x) * sp.special.comb(self.N - self.K, self.n - x) /
                sp.special.comb(self.N, self.n))

    def mean(self):
        return self.n * self.K / self.N

    def variance(self):
        return self.n * self.K * (self.N - self.K) * (self.N - self.n) / (self.N ** 2 * (self.N - 1))