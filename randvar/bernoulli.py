import numpy as np
from .random_variable import RandomVariable


class Bernoulli(RandomVariable):
    def __init__(self, p):
        super().__init__([0, 1], [1-p, p])
        self.probability = p

    def mean(self):
        return self.probability

    def variance(self):
        return self.probability * (1 - self.probability)
