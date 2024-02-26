import numpy as np
from .random_variable import RandomVariable


class Bernoulli(RandomVariable):
    def __init__(self, p):
        self.p = p
        super().__init__(np.array([0, 1]), np.array([1-p, p]))

    def mean(self):
        return self.p

    def variance(self):
        return self.p * (1 - self.p)
