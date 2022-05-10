import numpy as np

class GaussianMixture():
    def __init__(self, weights, gauss_states):

        self.weights = weights
        self.gauss_states = gauss_states