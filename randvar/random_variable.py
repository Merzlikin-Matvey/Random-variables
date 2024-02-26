import numpy as np


class RandomVariable:
    def __init__(self, *args):
        if len(args) == 2:
            self.values = np.array(args[0])
            self.probabilities = np.array(args[1])
        elif len(args) == 1:
            self.values = np.array(args[0].keys())
            self.probabilities = np.array(args[0].values())
        else:
            raise ValueError("You must pass 2 lists or 1 dictionary")

        if len(self.values) != len(self.probabilities):
            raise ValueError("The number of values and probabilities must be the same")

    def get_value(self):
        return np.random.choice(self.values, p=self.probabilities)

    def set_vales(self, values):
        self.values = values

    def set_probabilities(self, probabilities):
        self.probabilities = probabilities

    def expected_value(self):
        return np.sum(self.values * self.probabilities)

    def variance(self):
        return np.sum((self.values - self.expected_value())**2 * self.probabilities)

    def standard_deviation(self):
        return np.sqrt(self.variance())
