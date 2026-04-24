import numpy as np
import matplotlib.pyplot as plt
from parameters_separator import ParametersSeparator
from perceptron import Perceptron

'''
Cria os parâmetros necessários a partir da classe que separa os parametros do csv.
'''
M, X, W, y, x1, x2 = ParametersSeparator("spiral_d (1).csv", (0,1)).createParameters()
rosenblattPerceptron = Perceptron(X, M, W, y, 10000, 0.01)

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
    x2_space = (W[0] - W[1] * x1_space) / 2
    ax.plot(x1_space, x2_space, color='r', label='Fronteira')

plt.show()
