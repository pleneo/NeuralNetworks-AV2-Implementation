import numpy as np

class DataMonteCarlifier:
    def __init__(self, M):
        self.M = M

    def matrix_carlifier(self):
        M = self.M

        '''
        Adiciona o o vetor x_0 (-1) À matriz para que ela esteja completa:
        [bias, x1, x2, y]
        '''
        M = np.hstack((-np.ones((M.shape[0], 1)), M))

        np.random.shuffle(M)

        split_point = int(.8 * len(M))
        train_M, test_M = M[:split_point], M[split_point:]





        return train_M, test_M
