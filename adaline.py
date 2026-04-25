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



    def fit(self, max_epochs, learning_rate, precision):
        X_train = self.M[:, :2]
        X_train = np.hstack((-np.ones(X_train.shape[0]), X_train))
        y = self.M[:, -1]
        W = np.random.uniform(self.M.shape[1])

        epochs = 0
        eqm_atual = 5
        eqm_antiga = 1
        while epochs <= max_epochs or (eqm_atual - eqm_antiga) <= precision:
            #eqm recebe novo qm

            for i in range(self.M.shape[0]):
                u_k = np.dot(W, X_train)
                d_k = [i, -1]

                eqm = eqm + (d_k - u_k)**2

                W += learning_rate * (d_k - u_k) * X_train[i]

            epochs+=1

            eqm = 1/X_train.shape[0] * eqm



