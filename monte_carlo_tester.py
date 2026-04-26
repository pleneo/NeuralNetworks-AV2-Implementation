import numpy as np
from data_monte_carlifier_splitter import DataMonteCarlifier
from perceptron import Perceptron
from adaline import Adaline

def tests_set(M, R = 500, max_epochs = 10000, learning_rate = 0.0001, precision = 1e-6):
    accuracies = [[],[]]
    sensibilities = [[],[]]
    specificities = [[],[]]
    precisions = [[],[]]
    f1_scores = [[],[]]
    confusion_matrixes_perceptron = []
    confusion_matrixes_adaline = []

    for i in range(R):
        M_train, M_test = DataMonteCarlifier(M).matrix_carlifier()

        W = np.random.uniform(0, 1, M_train.shape[1]-1)

        rosenblattPerceptron = Perceptron(M_train[:, :3], M_train, W, M_train[:, -1], max_epochs, learning_rate)

        rosenblattPerceptron.fit()

        W_perceptron = rosenblattPerceptron.W


        adaline = Adaline(M_train[:, 1:])
        W_adaline = adaline.fit(max_epochs, learning_rate, precision)


        tester_perceptron = MonteCarloTester(M_test, W_perceptron)

        confusion_matrix_perceptron = tester_perceptron.run_test()
        confusion_matrixes_perceptron.append(confusion_matrix_perceptron)

        accuracy, sensibility, specificity, precision, f1_score = tester_perceptron.calcutate_validation_metrics(confusion_matrix_perceptron)

        accuracies[0].append(accuracy)
        sensibilities[0].append(sensibility)
        specificities[0].append(specificity)
        precisions[0].append(precision)
        f1_scores[0].append(f1_score)


        tester_adaline = MonteCarloTester(M_test, W_adaline)

        confusion_matrix_adaline = tester_adaline.run_test()
        confusion_matrixes_adaline.append(confusion_matrix_adaline)

        accuracy, sensibility, specificity, precision, f1_score = tester_adaline.calcutate_validation_metrics(confusion_matrix_adaline)

        accuracies[1].append(accuracy)
        sensibilities[1].append(sensibility)
        specificities[1].append(specificity)
        precisions[1].append(precision)
        f1_scores[1].append(f1_score)






    argmin_acc, argmax_acc = np.argmin(accuracies), np.argmax(accuracies)
    argmin_sens, argmax_sens = np.argmin(sensibilities), np.argmax(sensibilities)
    argmin_spec, argmax_spec = np.argmin(specificities), np.argmax(specificities)
    argmin_prec, argmax_prec = np.argmin(precisions), np.argmax(precisions)
    argmin_f1, argmax_f1 = np.argmin(f1_scores), np.argmax(f1_scores)

    # min_acc_cm, max_acc_cm = confusion_matrixes_perceptron[argmin_acc], confusion_matrixes_perceptron[argmax_acc]
    # min_sens_cm, max__cm = confusion_matrixes_perceptron[argmin_sens], confusion_matrixes_perceptron[argmax_sens]
    # min_spec_cm, max__cm = confusion_matrixes_perceptron[argmin_spec], confusion_matrixes_perceptron[argmax_spec]
    # min_prec_cm, max__cm = confusion_matrixes_perceptron[argmin_prec], confusion_matrixes_perceptron[argmax_prec]
    # min_f1_cm, max__cm = confusion_matrixes_perceptron[argmin_f1], confusion_matrixes_perceptron[argmax_f1]

#     min_acc, max_acc = np.min(accuracies), np.max(accuracies)
#     min_sens, max_sens = np.min(sensibilities), np.max(sensibilities)
#     min_spec, max_spec = np.min(specificities), np.max(specificities)
#     min_prec, max_prec = np.min(precisions), np.max(precisions)
#     min_f1, max_f1 = np.min(f1_scores), np.max(f1_scores)

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

        accuracy = (VP + VN) / (VP + VN + FP + FN)
        sensibility = VP / (VP + FN) if (VP + FN) != 0 else np.nan
        specificity = VN / (VN + FP) if (VN + FP) != 0 else np.nan
        precision = VP / (VP + FP) if (VP + FP) != 0 else 0.0

        f1_score = (2 * precision * sensibility) / (precision + sensibility) if (precision + sensibility) != 0 else 0.0

        return accuracy, sensibility, specificity, precision, f1_score

