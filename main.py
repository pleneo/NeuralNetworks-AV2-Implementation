import numpy as np
import matplotlib.pyplot as plt
from parameters_separator import ParametersSeparator
from perceptron import Perceptron
from data_monte_carlifier_splitter import DataMonteCarlifier
from monte_carlo_tester import MonteCarloTester, tests_set

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

# tester = MonteCarloTester(M_test, W)
#
# confusion_matrix = tester.run_test()
#
# accuracy, sensibility, specificity, precision, f1_score = tester.calcutate_validation_metrics(confusion_matrix)

'''
Na primeira vez que rodei, retornou:
0.7964285714285714 0.975 0.35 0.7894736842105263 0.43624161073825507

isso significa:
79.64% de acurácia, que a principio parece ser um valor interessante de acerto.
97.5% de sensiblidade,o que indica alto acerto de positivos (Verdadeiros Positivos / True Positives)
Mas a especificidade mostra o erro crasso do modelo (perceptron simples):
tester = MonteCarloTester(M_test, W)

confusion_matrix = tester.run_test()
 35% de acerto, ou seja, para valores negativos, houve altissima falha,
mostrando que o modelo está enviesado para valores positivos e só acerta eles devido maior densidade de amostras que retornam positivo.
Precisão de 78.94% demonstra que a maior parte dos positivos preditos realmente são positivos

Mas o f1-score mostra que a média harmônica entre a precisão e a sensibilidade mostra que o modelo foi teve baixa performance (43.62%),
mostrando que o modelo é péssimo em situaçaõ onde o resultante final é negativo.
'''
# print(accuracy,sensibility,specificity,precision,f1_score)

accuracies, sensibilities, specificities, precisions, f1_scores =  tests_set(M,10)

print(np.mean(accuracies))
print(np.mean(sensibilities))
print(np.mean(specificities))
print(np.mean(precisions))
print(np.mean(f1_scores))