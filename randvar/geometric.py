import numpy as np
from .random_variable import RandomVariable


class Geometric(RandomVariable):
    def __init__(self, p, max_value=100):
        self.p = p
        self.values = np.arange(1, max_value + 1)
        self.probabilities = np.array([(1 - p) for _ in range(len(self.values))]) ** (self.values - 1) * p
        self.probabilities[-1] = 1 - np.sum(self.probabilities[:-1])
        super().__init__(self.values, self.probabilities)

    def mean(self):
        return 1 / self.p

    def variance(self):
        return (1 - self.p) / (self.p ** 2)
