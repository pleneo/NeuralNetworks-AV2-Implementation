import numpy as np
from data_monte_carlifier_splitter import DataMonteCarlifier

class MonteCarloTester:
    def __init__(self, M_test, W):
        self.M_test = M_test
        self.W = W
        self.hits = 0 # accuracy

    def _bipolar_step_activation_function(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def run_test(self):
        X_test = self.M_test[:, :3]

        for i in range(X_test.shape[0]):
            u_k = np.dot(X_test[i], self.W)
            y_k = self._bipolar_step_activation_function(u_k)
            d_k = self.M_test[i, -1]
            if d_k == y_k:
                self.hits += 1

        accuracy = self.hits / X_test.shape[0]
        print(accuracy)

