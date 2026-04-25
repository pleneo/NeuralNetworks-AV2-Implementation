import numpy as np

class ParametersSeparator:
    def __init__(self, file_name: str, x_cols):
        self.file_name = file_name


    def createParameters(self):

        M = np.loadtxt(self.file_name, delimiter=',')
        """
        Crio a matrix X de entradas contendo as colunas 0 e 1 (duas primeiras). Essas duas colunas representam os x1 e x2
        de cada amostra. Dessa maneira, cada linha da da matrix Nx2 será o x1 e x2 de uma amostra.
        """
        X = np.loadtxt(self.file_name, delimiter=',', usecols=(0, 1))

        '''
            y é o vetor coluna referente a saída encontrada de cada amostra. Nesse conjunto de dados, as saídas estão no formato
            degrau bipolar, onde cada linha do vetor Nx1 é uma saída do conjunto de entradas daquela amostra. 
            '''
        y = np.loadtxt("spiral_d (1).csv", delimiter=',', usecols=-1)

        '''
        Transformar as colunas 0 e 1 (x1 e x2) em vetores coluna para ser possível apresentá-los de maneira apropriada.
    
        TODO: É realmente necesśario fazer isso? pois dá pra plotar sem, só fazendo X[:,0/1]. Ver se é necessário para outras partes também
    
        '''
        x1 = X[:, 0].reshape(-1, 1)
        x2 = X[:, 1].reshape(-1, 1)

        '''
        Define o vetor x0 como sendo um vetor de -1's.
        '''
        X_0 = -np.ones((X.shape[0], 1))


        '''
        Define o vetor X final que será usado para treinar o modelo.
        '''
        X_to_train = np.hstack((X_0, X))

        '''
        Randomiza criação dos pesos sinápticos como sendo valores equiprováveis entre 0 e 1.
        Deixa ele no formato vetor coluna para permitir cálculos matriciais.
        O primeiro valor (w_0) representa o thresold.

        Utilizo somente o x1 e x2 ou uso x0 também? Acho que uso x0 pois o thresold (w0) é associado ao x0.

        '''
        W = np.random.uniform(0, 1, X_to_train.shape[1])

        return M, X_to_train, W, y, x1, x2
