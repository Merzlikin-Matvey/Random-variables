import numpy as np
import scipy as sp
from .random_variable import RandomVariable


class Binomial(RandomVariable):
    def __init__(self, n, probability):
        self.n = n
        self.probability = np.array(probability)
        super().__init__(np.arange(n + 1), [self.pmf(i) for i in range(n + 1)])

    def pmf(self, k):
        return sp.special.comb(self.n, k) * pow(self.probability, k) * pow(1 - self.probability, self.n - k)

    def mean(self):
        return self.n * self.probability

    def variance(self):
        return self.n * self.probability * (1 - self.probability)

