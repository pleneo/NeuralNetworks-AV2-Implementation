from operator import truediv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from numpy.f2py.capi_maps import sign2map

'''
Crio a matrix X de entradas contendo as colunas 0 e 1 (duas primeiras). Essas duas colunas representam os x1 e x2
de cada amostra. Dessa maneira, cada linha da da matrix Nx2 será o x1 e x2 de uma amostra.
'''
X = np.loadtxt("spiral_d (1).csv", delimiter=',', usecols=(0, 1))

'''
Transformar as colunas 0 e 1 (x1 e x2) em vetores coluna para ser possível apresentá-los de maneira apropriada.

TODO: É realmente necesśario fazer isso? pois dá pra plotar sem, só fazendo X[:,0/1]. Ver se é necessário para outras partes também

'''
x1 = X[:,0].reshape(-1, 1)
x2 = X[:,1].reshape(-1, 1)
'''
y é o vetor coluna referente a saída encontrada de cada amostra. Nesse conjunto de dados, as saídas estão no formato
degrau bipolar, onde cada linha do vetor Nx1 é uma saída do conjunto de entradas daquela amostra. 
'''
y = np.loadtxt("spiral_d (1).csv", delimiter=',', usecols=2)

figure = plt.figure()
ax = figure.add_subplot()

plt.title("Gráfico de espalhamento para spiral_d")

ax.scatter(x1,x2, c="cyan", edgecolor='k')

# plt.show()

X_0 = -np.ones((X.shape[0],1))

X_to_train = np.hstack((X_0,X))

'''
Randomiza criação dos pesos sinápticos como sendo valores equiprováveis entre 0 e 1.
Deixa ele no formato vetor coluna para permitir cálculos matriciais.
O primeiro valor (w_0) representa o thresold.

Utilizo somente o x1 e x2 ou uso x0 também? Acho que uso x0 pois o thresold (w0) é associado ao x0.
 
'''
W = np.random.uniform(0,1, X_to_train.shape[1])

'''
Inicializa contador de épocas
'''
epochs = 0

learning_rate = 0.01

def degrau_bipolar(x):
    if x >= 0:
        return 1
    else:
        return -1


error = True

while error and epochs < 10000:
    error = False
    for x in range(X_to_train.shape[0]):
        x_k = X_to_train[x]
        u_k = np.dot(W, x_k)''
        y_k = degrau_bipolar(u_k)
        d_k = y[x]
        if(y_k != d_k):
            W = W + (learning_rate * (d_k - y_k) * x_k)
            error = True
    epochs+= 1

print(epochs)

x1_min, x1_max = X[:,0].min(), X[:, 0].max()
x1_space = np.linspace(x1_min, x1_max, 100)

if W[2] != 0:
    x2_space = (W[0] - W[1] * x1_space) / 2
    ax.plot(x1_space, x2_space, color='r', label='Fronteira')

plt.show()
a=1