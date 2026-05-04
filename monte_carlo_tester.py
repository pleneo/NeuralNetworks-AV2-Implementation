import numpy as np

from adaline import Adaline
from data_monte_carlifier_splitter import DataMonteCarlifier, normalize_train_test
from multilayered_perceptron import MultilayeredPerceptron
from perceptron import Perceptron


MODEL_KEYS = ("perceptron", "adaline", "mlp")
MODEL_LABELS = {
    "perceptron": "Perceptron Simples",
    "adaline": "ADALINE",
    "mlp": "MLP",
}

METRIC_KEYS = ("accuracy", "sensibility", "specificity", "precision", "f1_score")
METRIC_LABELS = {
    "accuracy": "Acurácia",
    "sensibility": "Sensibilidade",
    "specificity": "Especificidade",
    "precision": "Precisão",
    "f1_score": "F1-score",
}


class MonteCarloResults:
    def __init__(self):
        self.metrics = {
            model_key: {metric_key: [] for metric_key in METRIC_KEYS}
            for model_key in MODEL_KEYS
        }
        self.records = {model_key: [] for model_key in MODEL_KEYS}

    def add_record(self, model_key, round_index, confusion_matrix, metric_values, learning_curve):
        record = {
            "round": round_index,
            "confusion_matrix": np.asarray(confusion_matrix, dtype=int),
            "metrics": metric_values,
            "learning_curve": list(learning_curve),
        }
        self.records[model_key].append(record)

        for metric_key in METRIC_KEYS:
            self.metrics[model_key][metric_key].append(metric_values[metric_key])

    def summary(self, model_key, metric_key):
        values = np.asarray(self.metrics[model_key][metric_key], dtype=float)
        return {
            "mean": np.nanmean(values),
            "std": np.nanstd(values),
            "max": np.nanmax(values),
            "min": np.nanmin(values),
        }

    def best_worst_cases(self, model_key, metric_key):
        values = np.asarray(self.metrics[model_key][metric_key], dtype=float)
        best_index = int(np.nanargmax(values))
        worst_index = int(np.nanargmin(values))
        return {
            "best": self.records[model_key][best_index],
            "worst": self.records[model_key][worst_index],
        }

    def as_legacy_tuple(self):
        return tuple(
            [
                [self.metrics[model_key][metric_key] for model_key in MODEL_KEYS]
                for metric_key in METRIC_KEYS
            ]
        )


def tests_set(
    M,
    R=500,
    max_epochs=10000,
    learning_rate=0.0001,
    precision=1e-6,
    mlp_topology=(10,),
    mlp_learning_rate=1e-2,
    mlp_max_epochs=1000,
    mlp_precision=1e-6,
    perceptron_max_epochs=None,
    perceptron_learning_rate=None,
    adaline_max_epochs=None,
    adaline_learning_rate=None,
    return_legacy=False,
):
    perceptron_max_epochs = perceptron_max_epochs or max_epochs
    perceptron_learning_rate = perceptron_learning_rate or learning_rate
    adaline_max_epochs = adaline_max_epochs or max_epochs
    adaline_learning_rate = adaline_learning_rate or learning_rate

    results = MonteCarloResults()

    for round_index in range(1, R + 1):
        M_train, M_test = DataMonteCarlifier(M).matrix_carlifier()
        M_train, M_test = normalize_train_test(M_train, M_test)

        perceptron_weights = np.random.uniform(0, 1, M_train.shape[1] - 1)
        perceptron = Perceptron(
            M_train[:, :3],
            M_train,
            perceptron_weights,
            M_train[:, -1],
            perceptron_max_epochs,
            perceptron_learning_rate,
        )
        perceptron.fit()

        adaline = Adaline(M_train[:, 1:])
        adaline.fit(adaline_max_epochs, adaline_learning_rate, precision)

        X_train_mlp = M_train[:, 1:3].T
        Y_train_mlp = M_train[:, -1].reshape(1, -1)
        mlp = MultilayeredPerceptron(
            list(mlp_topology),
            X_train_mlp,
            Y_train_mlp,
            mlp_learning_rate,
            mlp_max_epochs,
            mlp_precision,
            normalize_inputs=False,
        )
        mlp.fit()

        evaluate_model(results, "perceptron", round_index, M_test, perceptron)
        evaluate_model(results, "adaline", round_index, M_test, adaline)
        evaluate_model(results, "mlp", round_index, M_test, mlp)

    if return_legacy:
        return results.as_legacy_tuple()

    return results


def evaluate_model(results, model_key, round_index, M_test, model):
    tester = MonteCarloTester(M_test, model)
    confusion_matrix = tester.run_test()
    metric_values = tester.calculate_validation_metrics(confusion_matrix)
    learning_curve = getattr(model, "learning_curve", [])
    results.add_record(model_key, round_index, confusion_matrix, metric_values, learning_curve)


class MonteCarloTester:
    def __init__(self, M_test, model_or_weights):
        self.M_test = M_test
        self.model_or_weights = model_or_weights

    def _bipolar_step_activation_function(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def run_test(self):
        X_test = self.M_test[:, :3]
        confusion_matrix = [[0, 0], [0, 0]]

        for i in range(X_test.shape[0]):
            y_k = self._predict_sample(X_test[i])
            d_k = self.M_test[i, -1]

            if d_k == y_k and (d_k == +1 and y_k == +1):
                confusion_matrix[0][0] += 1 # VP

            if d_k == y_k and (d_k == -1 and y_k == -1):
                confusion_matrix[1][1] += 1 # VN

            if d_k != y_k and (d_k == -1 and y_k == +1):
                confusion_matrix[0][1] += 1 # FP

            if d_k != y_k and (d_k == +1 and y_k == -1):
                confusion_matrix[1][0] += 1 # FN

        return confusion_matrix

    def _predict_sample(self, x_with_bias):
        if hasattr(self.model_or_weights, "predict"):
            if isinstance(self.model_or_weights, MultilayeredPerceptron):
                prediction = self.model_or_weights.predict(x_with_bias[1:])
            else:
                prediction = self.model_or_weights.predict(x_with_bias)
            return int(np.asarray(prediction).reshape(-1)[0])

        u_k = np.dot(x_with_bias, self.model_or_weights)
        return self._bipolar_step_activation_function(u_k)

    def calculate_validation_metrics(self, confusion_matrix):
        VP = confusion_matrix[0][0]
        VN = confusion_matrix[1][1]
        FP = confusion_matrix[0][1]
        FN = confusion_matrix[1][0]

        accuracy = (VP + VN) / (VP + VN + FP + FN)
        sensibility = VP / (VP + FN) if (VP + FN) != 0 else np.nan
        specificity = VN / (VN + FP) if (VN + FP) != 0 else np.nan
        precision = VP / (VP + FP) if (VP + FP) != 0 else 0.0
        f1_score = (
            (2 * precision * sensibility) / (precision + sensibility)
            if (precision + sensibility) != 0
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "sensibility": sensibility,
            "specificity": specificity,
            "precision": precision,
            "f1_score": f1_score,
        }

    def calcutate_validation_metrics(self, confusion_matrix):
        metrics = self.calculate_validation_metrics(confusion_matrix)
        return tuple(metrics[metric_key] for metric_key in METRIC_KEYS)
