import numpy as np


class MultilayeredPerceptron:
    def __init__(
        self,
        topology,
        X_train,
        Y_train,
        learning_rate,
        max_epochs,
        precision,
        normalize_inputs=True,
    ):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.precision = precision
        self.normalize_inputs = normalize_inputs
        self.learning_curve = []

        self.p, self.N = X_train.shape
        self.output_dim = Y_train.shape[0]
        self.topology = list(topology) + [self.output_dim]

        if normalize_inputs:
            self.feature_min = X_train.min(axis=1, keepdims=True)
            self.feature_max = X_train.max(axis=1, keepdims=True)
            self.feature_range = np.where(
                self.feature_max - self.feature_min == 0,
                1,
                self.feature_max - self.feature_min,
            )
        else:
            self.feature_min = np.zeros((self.p, 1))
            self.feature_range = np.ones((self.p, 1))

        X_scaled = self._scale_inputs(X_train)
        self.X_train = np.vstack((-np.ones((1, self.N)), X_scaled))
        self.D = Y_train

        self.W = []
        input_dim = self.p + 1
        for neurons in self.topology:
            W = np.random.random_sample((neurons, input_dim)) - 0.5
            self.W.append(W)
            input_dim = neurons + 1

        self.u = [None] * len(self.W)
        self.y = [None] * len(self.W)
        self.delta = [None] * len(self.W)

    def _scale_inputs(self, X):
        if not self.normalize_inputs:
            return X
        return 2 * ((X - self.feature_min) / self.feature_range) - 1

    def g(self, u):
        return np.tanh(u / 2.0)

    def g_d(self, u):
        s = self.g(u)
        return 0.5 * (1 - s**2)

    def eqm(self):
        eqm = 0.0
        for k in range(self.N):
            x_k = self.X_train[:, k].reshape(self.p + 1, 1)
            self.forward(x_k)
            d_k = self.D[:, k].reshape(self.output_dim, 1)
            eqm += np.sum((d_k - self.y[-1]) ** 2)
        return eqm / (2 * self.N)

    def forward(self, x):
        for j, W in enumerate(self.W):
            if j == 0:
                self.u[j] = W @ x
            else:
                yb = np.vstack((-np.ones((1, 1)), self.y[j - 1]))
                self.u[j] = W @ yb
            self.y[j] = self.g(self.u[j])
        return self.y[-1]

    def backward(self, x, d):
        for j in range(len(self.W) - 1, -1, -1):
            if j == len(self.W) - 1:
                self.delta[j] = self.g_d(self.u[j]) * (d - self.y[j])
            else:
                Wnb = self.W[j + 1][:, 1:].T
                self.delta[j] = self.g_d(self.u[j]) * (Wnb @ self.delta[j + 1])

            if j == 0:
                input_with_bias = x
            else:
                input_with_bias = np.vstack((-np.ones((1, 1)), self.y[j - 1]))

            self.W[j] = self.W[j] + self.learning_rate * self.delta[j] @ input_with_bias.T

    def fit(self):
        epochs = 0
        eqm = self.eqm()
        self.learning_curve = [eqm]

        while epochs < self.max_epochs and eqm > self.precision:
            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p + 1, 1)
                d_k = self.D[:, k].reshape(self.output_dim, 1)
                self.forward(x_k)
                self.backward(x_k, d_k)

            epochs += 1
            eqm = self.eqm()
            self.learning_curve.append(eqm)

        return self

    def predict_raw(self, x):
        x = np.asarray(x, dtype=float).reshape(self.p, 1)
        x_scaled = self._scale_inputs(x)
        x_with_bias = np.vstack((-np.ones((1, 1)), x_scaled))
        return self.forward(x_with_bias)

    def predict_raw_batch(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.p:
            raise ValueError(f"Expected {self.p} features, received {X.shape[1]}")

        X_scaled = self._scale_inputs(X.T)
        activations = np.vstack((-np.ones((1, X_scaled.shape[1])), X_scaled))
        for layer_index, W in enumerate(self.W):
            u = W @ activations
            y = self.g(u)
            if layer_index < len(self.W) - 1:
                activations = np.vstack((-np.ones((1, y.shape[1])), y))
            else:
                activations = y
        return activations

    def predict(self, x):
        raw_output = self.predict_raw(x)
        return np.where(raw_output >= 0, 1, -1).reshape(-1)

    def predict_batch(self, X):
        raw_output = self.predict_raw_batch(X)
        return np.where(raw_output >= 0, 1, -1).reshape(-1)
