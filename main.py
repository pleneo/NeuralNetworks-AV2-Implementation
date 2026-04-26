import numpy as np
import matplotlib.pyplot as plt
from parameters_separator import ParametersSeparator
from perceptron import Perceptron
from data_monte_carlifier_splitter import DataMonteCarlifier
from monte_carlo_tester import MonteCarloTester, tests_set
from adaline import Adaline

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

adaline = Adaline(M_train[:, 1:])
W_adaline = adaline.fit(100000, .00001, 1e-6)

if W_adaline[2] != 0:
    x2_adaline_space = (W_adaline[0] - W_adaline[1] * x1_space) / W_adaline[2]
    ax.plot(x1_space, x2_adaline_space, color='b', label='Fronteira')


plt.show()












'''
Agora vem os testes.
acurácia: nº de acertos / nº de casos de teste.
'''

tester_perceptron = MonteCarloTester(M_test, W)

confusion_matrix_perceptron = tester_perceptron.run_test()

accuracy, sensibility, specificity, precision, f1_score = tester_perceptron.calcutate_validation_metrics(confusion_matrix_perceptron)

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

accuracies, sensibilities, specificities, precisions, f1_scores =  tests_set(M, 1, 11, .001)

print("\n\n ------- PERCEPTRON ------ \n\n")


print(np.mean(accuracies[0]))
print(np.mean(sensibilities[0]))
print(np.mean(specificities[0]))
print(np.mean(precisions[0]))
print(np.mean(f1_scores[0]))


print("\n\n ------- ADALINE ------ \n\n")

print("METRICS REACHED: ")
print(np.mean(accuracies[1]))
print(np.mean(sensibilities[1]))
print(np.mean(specificities[1]))
print(np.mean(precisions[1]))
print(np.mean(f1_scores[1]))

