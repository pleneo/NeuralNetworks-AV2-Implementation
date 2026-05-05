import numpy as np


class Adaline:
    def __init__(self, M):
        self.M = M
        self.W = None
        self.learning_curve = []
        self.progress_prefix = ""
        self.halfway_epoch = 0

    def _bipolar_step_activation_function(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def _calculate_least_mean_squared(self, p, W, X_train):
        predictions = X_train @ W
        targets = self.M[:, -1]
        errors = targets - predictions
        return np.sum(errors**2) / p

    def fit(self, max_epochs, learning_rate, precision):
        X_train = self.M[:, :2]
        ones = -np.ones(X_train.shape[0]).reshape(-1, 1)
        X_train = np.hstack((ones, X_train))
        W = np.random.uniform(0, 1, self.M.shape[1])
        self.halfway_epoch = max_epochs // 2

        epochs = 0
        isPrecisionReached = False
        eqm_atual = self._calculate_least_mean_squared(self.M.shape[0], W, X_train)
        self.learning_curve = [eqm_atual]

        while epochs < max_epochs and not isPrecisionReached:
            eqm_antiga = eqm_atual

            for i in range(self.M.shape[0]):
                u_k = np.dot(W, X_train[i])
                d_k = self.M[i, -1]

                W = W + learning_rate * (d_k - u_k) * X_train[i]

            epochs+=1

            eqm_atual = self._calculate_least_mean_squared(self.M.shape[0], W, X_train)
            self.learning_curve.append(eqm_atual)
            if self.halfway_epoch > 0 and epochs == self.halfway_epoch:
                print(f"{self.progress_prefix}[Adaline] epoca {epochs} | eqm={eqm_atual:.8f}")

            if abs(eqm_atual - eqm_antiga) <= precision:
                # print("reached precision ", precision)
                # print(eqm_atual - eqm_antiga)
                isPrecisionReached = True

        # if epochs >= max_epochs:
            # print(epochs)
            # print(abs(eqm_atual - eqm_antiga))
            # print("Converged by reaching max epochs")

        # if isPrecisionReached:
        #     # print(abs(eqm_atual - eqm_antiga))
        #     # print("Converged by reaching the desired precision")

        self.W = W
        return W

    def predict(self, x_with_bias):
        u = np.dot(self.W, x_with_bias)
        return self._bipolar_step_activation_function(u)

    def predict_batch(self, X_with_bias):
        activations = np.asarray(X_with_bias) @ self.W
        return np.where(activations >= 0, 1, -1)
