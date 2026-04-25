import numpy as np
import matplotlib.pyplot as plt
from parameters_separator import ParametersSeparator
from perceptron import Perceptron
from data_monte_carlifier_splitter import DataMonteCarlifier
from monte_carlo_tester import MonteCarloTester
'''
Cria os parâmetros necessários a partir da classe que separa os parametros do csv.
'''
M, X, W, y, x1, x2 = ParametersSeparator("spiral_d (1).csv", (0,1)).createParameters()


'''
80% são dados de treino (M_train) e os 20% restantes são para testes (M_test).
Retorna as matrizes contendo o bias,
'''
M_train, M_test = DataMonteCarlifier(M).matrix_carlifier()

rosenblattPerceptron = Perceptron(M_train[:, :3], M_train, W, M_train[:,-1], 100, .5)

figure = plt.figure()
ax = figure.add_subplot()

plt.title("Gráfico de espalhamento para spiral_d")

ax.scatter(x1,x2, c="cyan", edgecolor='k')

# plt.show()

rosenblattPerceptron.fit()
W = rosenblattPerceptron.W

x1_min, x1_max = X[:,1].min(), X[:, 1].max()
x1_space = np.linspace(x1_min, x1_max, 100)

if W[2] != 0:
    x2_space = (W[0] - W[1] * x1_space) / W[2]
    ax.plot(x1_space, x2_space, color='r', label='Fronteira')

plt.show()

'''
Agora vem os testes.
acurácia: nº de acertos / nº de casos de teste.
'''

tester = MonteCarloTester(M_test, W)

tester.run_test()
