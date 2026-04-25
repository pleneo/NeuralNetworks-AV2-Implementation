import numpy as np
from data_monte_carlifier_splitter import DataMonteCarlifier
from perceptron import Perceptron


def tests_set(M, R = 500):
    accuracies = []
    sensibilities = []
    specificities = []
    precisions = []
    f1_scores = []

    for i in range(R):
        W = np.random.uniform(0, 1, M.shape[1])

        M_train, M_test = DataMonteCarlifier(M).matrix_carlifier()

        rosenblattPerceptron = Perceptron(M_train[:, :3], M_train, W, M_train[:, -1], 100, .5)

        rosenblattPerceptron.fit()

        W = rosenblattPerceptron.W

        tester = MonteCarloTester(M_test, W)

        confusion_matrix = tester.run_test()

        accuracy, sensibility, specificity, precision, f1_score = tester.calcutate_validation_metrics(confusion_matrix)
        accuracies.append(accuracy)
        sensibilities.append(sensibility)
        specificities.append(specificity)
        precisions.append(precision)
        f1_scores.append(f1_score)

    return accuracies, sensibilities, specificities, precisions, f1_scores


class MonteCarloTester:
    def __init__(self, M_test, W):
        self.M_test = M_test
        self.W = W

    def _bipolar_step_activation_function(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def run_test(self):
        X_test = self.M_test[:, :3]
        confusion_matrix = [[0,0],[0,0]]

        for i in range(X_test.shape[0]):
            u_k = np.dot(X_test[i], self.W)
            y_k = self._bipolar_step_activation_function(u_k)
            d_k = self.M_test[i, -1]

            if d_k == y_k and (d_k == +1 and y_k == +1):
                confusion_matrix[0][0]+=1 #VP

            if d_k == y_k and (d_k == -1 and y_k == -1):
                confusion_matrix[1][1]+=1 #VN

            if d_k != y_k and (d_k == -1 and y_k == +1):
                confusion_matrix[0][1]+=1 #FP

            if d_k != y_k and (d_k == +1 and y_k == -1):
                confusion_matrix[1][0]+=1 #FN

        return confusion_matrix

    '''
    Calcula as métricas de validação a partir da matriz de confusão retornada de run_test().
    Acurácia: quantas vezes o modelo acertou, dividido pelo total de amostras disponíveis.
    Sensibilidade: Mede a razão entre os valores positivos (+1) corretamente identificados dividido pelo total de valores identificados corretamente como positivo e incorretamente como negativo (que na real era positivo).
    Especificidade: Mesma lógica da sensibilidade, mas para os negativos. Mede os corretamente identificados como negativo dividio pelo total de valores identificados corretamente como negativo e incorretamente como positivo (que na real era negativo).
    Precisão: Proporção de valores identificados como positivo pelo total de positivos (verdadeiros e falsos)
    F1-Score: Média harmônica entre precisão e sensibilidade, equilibrando as duas métricas. 
    '''
    def calcutate_validation_metrics(self, confusion_matrix):
        VP = confusion_matrix[0][0]
        VN = confusion_matrix[1][1]
        FP = confusion_matrix[0][1]
        FN = confusion_matrix[1][0]

        accuracy = (VP + VN)/(VP+VN+FP+FN)
        sensibility = VP/(VP+FN)
        specificity = VN/(VN+FP)
        precision = VP/(VP+FP)

        f1_score = (precision * sensibility) / (precision + sensibility)

        return accuracy, sensibility, specificity, precision, f1_score

