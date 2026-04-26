import numpy as np


class Adaline:
    def __init__(self, M):
        self.M = M

    def _bipolar_step_activation_function(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def _calculate_least_mean_squared(self, p, W, X_train):
        eqm = 0
        for i in range(X_train.shape[0]):
            u = np.dot(W, X_train[i])
            d_k = self.M[i, -1]
            eqm = eqm + ((d_k - u)**2)

        eqm = eqm/p

        return eqm

    def fit(self, max_epochs, learning_rate, precision):
        eqm_atual = 0
        eqm_antiga = 0
        X_train = self.M[:, :2]
        ones = -np.ones(X_train.shape[0]).reshape(-1, 1)
        X_train = np.hstack((ones, X_train))
        y = self.M[:, -1]
        W = np.random.uniform(0, 1, self.M.shape[1])

        epochs = 0
        isPrecisionReached = False

        while epochs <= max_epochs and not isPrecisionReached:
            eqm_antiga = self._calculate_least_mean_squared(self.M.shape[0], W, X_train)

            for i in range(self.M.shape[0]):
                u_k = np.dot(W, X_train[i])
                d_k = self.M[i, -1]

                W = W + learning_rate * (d_k - u_k) * X_train[i]

            epochs+=1

            eqm_atual = self._calculate_least_mean_squared(self.M.shape[0], W, X_train)

            if abs(eqm_atual - eqm_antiga) <= precision:
                print("reached precision ", precision)
                print(eqm_atual - eqm_antiga)
                isPrecisionReached = True

        if epochs >= max_epochs:
            print(epochs)
            print(abs(eqm_atual - eqm_antiga))
            print("Converged by reaching max epochs")

        if isPrecisionReached:
            print(abs(eqm_atual - eqm_antiga))
            print("Converged by reaching the desired precision")

        return W
