import numpy as np

class Perceptron:
    def __init__(self,X,M,W,y, max_epochs = 10000, learning_rate = 0.01):
        self.X = X
        self.M = M
        self.W = W
        self.y = y
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate

    def _degrau_bipolar(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def fit(self):
        epochs = 0

        learning_rate = self.learning_rate

        error = True

        while error and epochs < self.max_epochs:
            error = False
            for x in range(self.X.shape[0]):
                x_k = self.X[x]
                u_k = np.dot(self.W, x_k)
                y_k = self._degrau_bipolar(u_k)
                d_k = self.y[x]
                if y_k != d_k:
                    self.W = self.W + (learning_rate * (d_k - y_k) * x_k)
                    error = True
            epochs += 1


        print("Convergence reached by max epochs" if epochs == self.max_epochs else "Convergence reached in " + epochs + " epochs")

