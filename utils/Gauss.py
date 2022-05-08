import numpy as np

class GaussState:
    def __init__(self, mean, cov):

        self.mean = np.asarray(mean, dtype=np.float32).reshape((-1, 1))
        self.cov = np.asarray(cov, dtype=np.float32)
    

    ## This function allows for tuple unpacking
    def __iter__(self):
        return iter((self.mean, self.cov))
