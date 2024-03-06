import numpy as np
import scipy as sp
from .random_variable import RandomVariable


class Poisson(RandomVariable):
    def __init__(self, mu, minimal_value=None, maximal_value=None):
        self.mu = mu
        if (minimal_value is None) or (maximal_value is None):
            minimal_value = max(0, int(self.mu - 4 * np.sqrt(self.mu)))
            maximal_value = int(self.mu + 4 * np.sqrt(self.mu))

        self.values = np.arange(minimal_value, maximal_value + 1)
        self.probabilities = np.array([self.pmf(x) for x in self.values])
        super().__init__(self.values, self.probabilities)

    def pmf(self, x):
        return np.exp(-self.mu) * self.mu ** x / sp.special.factorial(x)

    def mean(self):
        return self.mu

    def variance(self):
        return self.mu
