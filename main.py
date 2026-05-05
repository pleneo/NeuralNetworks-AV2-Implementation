from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaline import Adaline
from data_monte_carlifier_splitter import DataMonteCarlifier, normalize_train_test
from monte_carlo_tester import (
    METRIC_KEYS,
    METRIC_LABELS,
    MODEL_KEYS,
    MODEL_LABELS,
    MonteCarloTester,
    tests_set,
)
from multilayered_perceptron import MultilayeredPerceptron
from parameters_separator import ParametersSeparator
from perceptron import Perceptron


DATA_FILE = "spiral_d (1).csv"
OUTPUT_DIR = Path("results")

PERCEPTRON_MAX_EPOCHS = 10000
PERCEPTRON_LEARNING_RATE = 1e-2

ADALINE_MAX_EPOCHS = 10000
ADALINE_LEARNING_RATE = 1e-2
ADALINE_PRECISION = 1e-8

MLP_TOPOLOGY = (10,)
MLP_LEARNING_RATE = 1e-2
MLP_MAX_EPOCHS = 10000
MLP_PRECISION = 1e-8

MONTE_CARLO_ROUNDS = 500
MLP_TOPOLOGY_STUDY = {
    "underfitting": (1,),
    "baseline": MLP_TOPOLOGY,
    "overfitting": (50, 50),
}


def load_spiral_data(file_name=DATA_FILE):
    M, *_ = ParametersSeparator(file_name, (0, 1)).createParameters()
    return M


def create_train_test_split(M):
    M_train, M_test = DataMonteCarlifier(M).matrix_carlifier()
    return normalize_train_test(M_train, M_test)


def plot_initial_scatter(M, ax):
    negative_class = M[:, -1] == -1
    positive_class = M[:, -1] == 1

    ax.scatter(
        M[negative_class, 0],
        M[negative_class, 1],
        c="tab:blue",
        edgecolor="k",
        label="Classe -1",
        alpha=0.8,
    )
    ax.scatter(
        M[positive_class, 0],
        M[positive_class, 1],
        c="tab:orange",
        edgecolor="k",
        label="Classe +1",
        alpha=0.8,
    )

    ax.set_title("Gráfico de espalhamento para spiral_d")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()


def plot_linear_decision_boundary(ax, W, x1_values, label, color):
    if W[2] == 0:
        return

    x2_values = (W[0] - W[1] * x1_values) / W[2]
    ax.plot(x1_values, x2_values, color=color, label=label)
    ax.legend()


def train_perceptron(M_train):
    initial_weights = np.random.uniform(0, 1, M_train.shape[1] - 1)
    model = Perceptron(
        M_train[:, :3],
        M_train,
        initial_weights,
        M_train[:, -1],
        PERCEPTRON_MAX_EPOCHS,
        PERCEPTRON_LEARNING_RATE,
    )
    model.fit()
    return model


def train_adaline(M_train):
    model = Adaline(M_train[:, 1:])
    model.fit(ADALINE_MAX_EPOCHS, ADALINE_LEARNING_RATE, ADALINE_PRECISION)
    return model


def train_mlp(M_train, topology=MLP_TOPOLOGY):
    X_train = M_train[:, 1:3].T
    Y_train = M_train[:, -1].reshape(1, -1)

    model = MultilayeredPerceptron(
        topology,
        X_train,
        Y_train,
        learning_rate=MLP_LEARNING_RATE,
        max_epochs=MLP_MAX_EPOCHS,
        precision=MLP_PRECISION,
        normalize_inputs=False,
    )
    model.fit()
    return model


def save_training_example(M, output_dir):
    M_train, M_test = create_train_test_split(M)
    M_normalized = np.vstack((M_train, M_test))[:, 1:]

    perceptron = train_perceptron(M_train)
    adaline = train_adaline(M_train)
    mlp = train_mlp(M_train)

    figure = plt.figure(figsize=(8, 6))
    ax = figure.add_subplot()
    plot_initial_scatter(M_normalized, ax)
    ax.set_title("Dados normalizados e fronteiras lineares")

    x1_min = M_normalized[:, 0].min()
    x1_max = M_normalized[:, 0].max()
    x1_values = np.linspace(x1_min, x1_max, 100)

    plot_linear_decision_boundary(
        ax,
        perceptron.W,
        x1_values,
        "Fronteira Perceptron",
        "tab:red",
    )
    plot_linear_decision_boundary(
        ax,
        adaline.W,
        x1_values,
        "Fronteira ADALINE",
        "tab:green",
    )

    save_figure(figure, output_dir / "01_dados_normalizados_fronteiras.png")

    figure = plt.figure(figsize=(8, 6))
    ax = figure.add_subplot()
    plot_mlp_decision_boundary(ax, mlp, M_normalized)
    plot_initial_scatter(M_normalized, ax)
    ax.set_title("Fronteira de decisão da MLP")
    save_figure(figure, output_dir / "02_fronteira_decisao_mlp.png")


def plot_mlp_decision_boundary(ax, mlp, M_normalized, grid_size=250):
    x_min, x_max = M_normalized[:, 0].min(), M_normalized[:, 0].max()
    y_min, y_max = M_normalized[:, 1].min(), M_normalized[:, 1].max()
    margin = 0.05

    xx, yy = np.meshgrid(
        np.linspace(x_min - margin, x_max + margin, grid_size),
        np.linspace(y_min - margin, y_max + margin, grid_size),
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.predict_batch(grid_points)
    zz = predictions.reshape(xx.shape)

    ax.contourf(
        xx,
        yy,
        zz,
        levels=[-1, 0, 1],
        colors=["tab:blue", "tab:orange"],
        alpha=0.18,
    )
    ax.contour(xx, yy, zz, levels=[0], colors="black", linewidths=1.5)
    ax.set_xlabel("x1 normalizado")
    ax.set_ylabel("x2 normalizado")


def run_monte_carlo_validation(M):
    return tests_set(
        M,
        R=MONTE_CARLO_ROUNDS,
        precision=ADALINE_PRECISION,
        mlp_topology=MLP_TOPOLOGY,
        mlp_learning_rate=MLP_LEARNING_RATE,
        mlp_max_epochs=MLP_MAX_EPOCHS,
        mlp_precision=MLP_PRECISION,
        perceptron_max_epochs=PERCEPTRON_MAX_EPOCHS,
        perceptron_learning_rate=PERCEPTRON_LEARNING_RATE,
        adaline_max_epochs=ADALINE_MAX_EPOCHS,
        adaline_learning_rate=ADALINE_LEARNING_RATE,
    )


def print_validation_results(results):
    for metric_key in METRIC_KEYS:
        print(f"\n\n------- {METRIC_LABELS[metric_key].upper()} -------")
        print(f"{'Modelo':<28} {'Média':>10} {'Desvio':>10} {'Maior':>10} {'Menor':>10}")

        for model_key in MODEL_KEYS:
            summary = results.summary(model_key, metric_key)
            print(
                f"{MODEL_LABELS[model_key]:<28} "
                f"{summary['mean']:>10.4f} "
                f"{summary['std']:>10.4f} "
                f"{summary['max']:>10.4f} "
                f"{summary['min']:>10.4f}"
            )


def write_summary_tables(results, output_dir):
    for metric_key in METRIC_KEYS:
        table_path = output_dir / f"tabela_{metric_key}.csv"
        with table_path.open("w", encoding="utf-8") as file:
            file.write("Modelo,Media,Desvio-Padrao,Maior Valor,Menor Valor\n")
            for model_key in MODEL_KEYS:
                summary = results.summary(model_key, metric_key)
                file.write(
                    f"{MODEL_LABELS[model_key]},"
                    f"{summary['mean']:.6f},"
                    f"{summary['std']:.6f},"
                    f"{summary['max']:.6f},"
                    f"{summary['min']:.6f}\n"
                )


def save_metric_boxplots(results, output_dir):
    for metric_key in METRIC_KEYS:
        figure, ax = plt.subplots(figsize=(8, 5))
        values = [results.metrics[model_key][metric_key] for model_key in MODEL_KEYS]
        labels = [MODEL_LABELS[model_key] for model_key in MODEL_KEYS]

        ax.boxplot(values, tick_labels=labels)
        ax.set_title(f"Distribuição - {METRIC_LABELS[metric_key]}")
        ax.set_ylabel(METRIC_LABELS[metric_key])
        ax.grid(axis="y", alpha=0.3)
        figure.autofmt_xdate(rotation=15)

        save_figure(figure, output_dir / f"boxplot_{metric_key}.png")


def save_best_worst_artifacts(results, output_dir):
    artifacts_dir = output_dir / "melhores_piores"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for model_key in MODEL_KEYS:
        for metric_key in METRIC_KEYS:
            cases = results.best_worst_cases(model_key, metric_key)
            for case_name, record in cases.items():
                prefix = f"{model_key}_{metric_key}_{case_name}_rodada_{record['round']}"
                metric_value = record["metrics"][metric_key]
                title = (
                    f"{MODEL_LABELS[model_key]} - {case_name} "
                    f"{METRIC_LABELS[metric_key]} = {metric_value:.4f}"
                )

                save_confusion_matrix(
                    record["confusion_matrix"],
                    title,
                    artifacts_dir / f"{prefix}_matriz_confusao.png",
                )
                save_learning_curve(
                    record["learning_curve"],
                    model_key,
                    title,
                    artifacts_dir / f"{prefix}_curva_aprendizado.png",
                )


def save_confusion_matrix(confusion_matrix, title, path):
    figure, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(confusion_matrix, cmap="Blues")
    figure.colorbar(image, ax=ax)

    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred +1", "Pred -1"])
    ax.set_yticklabels(["Real +1", "Real -1"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(confusion_matrix[i, j]), ha="center", va="center")

    save_figure(figure, path)


def save_learning_curve(learning_curve, model_key, title, path):
    figure, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(learning_curve)), learning_curve)
    ax.set_title(title)
    ax.set_xlabel("Época")
    ax.set_ylabel("Erros" if model_key == "perceptron" else "EQM")
    ax.grid(alpha=0.3)
    save_figure(figure, path)


def run_mlp_topology_study(M, output_dir):
    study_dir = output_dir / "mlp_topologias"
    study_dir.mkdir(parents=True, exist_ok=True)

    M_train, M_test = create_train_test_split(M)
    table_path = study_dir / "resumo_topologias_mlp.csv"

    with table_path.open("w", encoding="utf-8") as file:
        file.write("Caso,Topologia,Acuracia,Sensibilidade,Especificidade,Precisao,F1-score\n")

        for case_name, topology in MLP_TOPOLOGY_STUDY.items():
            mlp = train_mlp(M_train, topology)
            tester = MonteCarloTester(M_test, mlp)
            confusion_matrix = np.asarray(tester.run_test(), dtype=int)
            metrics = tester.calculate_validation_metrics(confusion_matrix)

            file.write(
                f"{case_name},{topology},"
                f"{metrics['accuracy']:.6f},"
                f"{metrics['sensibility']:.6f},"
                f"{metrics['specificity']:.6f},"
                f"{metrics['precision']:.6f},"
                f"{metrics['f1_score']:.6f}\n"
            )

            title = f"MLP {case_name} - topologia {topology}"
            save_confusion_matrix(
                confusion_matrix,
                title,
                study_dir / f"{case_name}_matriz_confusao.png",
            )
            save_learning_curve(
                mlp.learning_curve,
                "mlp",
                title,
                study_dir / f"{case_name}_curva_aprendizado.png",
            )
            save_mlp_topology_decision_boundary(
                mlp,
                M_train,
                M_test,
                title,
                study_dir / f"{case_name}_fronteira_decisao.png",
            )


def save_mlp_topology_decision_boundary(mlp, M_train, M_test, title, path):
    M_normalized = np.vstack((M_train, M_test))[:, 1:]

    figure, ax = plt.subplots(figsize=(8, 6))
    plot_mlp_decision_boundary(ax, mlp, M_normalized)
    plot_initial_scatter(M_normalized, ax)
    ax.set_title(title)
    save_figure(figure, path)


def save_figure(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    M = load_spiral_data()

    save_training_example(M, OUTPUT_DIR)

    results = run_monte_carlo_validation(M)
    print_validation_results(results)
    write_summary_tables(results, OUTPUT_DIR)
    save_metric_boxplots(results, OUTPUT_DIR)
    save_best_worst_artifacts(results, OUTPUT_DIR)
    run_mlp_topology_study(M, OUTPUT_DIR)

    print(f"\nArtefatos salvos em: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
