import numpy as np

'''
Cria o perceptron, passando como parametro a matriz de entrada de treino (bias, x1,x2), o vetor W de pesos, o vetor y de resultados,
o max_epochs e o learning rate. Possui o método fit() que treina o modelo.
'''
class Perceptron:
    def __init__(self,X,M,W,y, max_epochs = 10000, learning_rate = 0.01):
        self.X = X
        self.M = M
        self.W = W
        self.y = y
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.learning_curve = []

    def _bipolar_step_activation_function(self, x):
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
            epoch_errors = 0
            for x in range(self.X.shape[0]):
                x_k = self.X[x]
                u_k = np.dot(self.W, x_k)
                y_k = self._bipolar_step_activation_function(u_k)
                d_k = self.y[x]
                if y_k != d_k:
                    self.W = self.W + (learning_rate * (d_k - y_k) * x_k)
                    error = True
                    epoch_errors += 1
            epochs += 1
            self.learning_curve.append(epoch_errors)


        #print("Convergence reached by max epochs" if epochs == self.max_epochs else 'hi lorena')
        return self

    def predict(self, x_with_bias):
        u = np.dot(self.W, x_with_bias)
        return self._bipolar_step_activation_function(u)

    def predict_batch(self, X_with_bias):
        activations = np.asarray(X_with_bias) @ self.W
        return np.where(activations >= 0, 1, -1)
