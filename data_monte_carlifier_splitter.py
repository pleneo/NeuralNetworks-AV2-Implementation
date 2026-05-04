import numpy as np

class DataMonteCarlifier:
    def __init__(self, M):
        self.M = M

    def matrix_carlifier(self):
        M = self.M.copy()

        '''
        Adiciona o o vetor x_0 (-1) À matriz para que ela esteja completa:
        [bias, x1, x2, y]
        '''
        M = np.hstack((-np.ones((M.shape[0], 1)), M))

        np.random.shuffle(M)

        split_point = int(.8 * len(M))
        train_M, test_M = M[:split_point], M[split_point:]





        return train_M, test_M


def normalize_train_test(M_train, M_test):
    M_train_norm = M_train.copy()
    M_test_norm = M_test.copy()

    train_features = M_train[:, 1:3]
    feature_min = train_features.min(axis=0)
    feature_max = train_features.max(axis=0)
    feature_range = np.where(feature_max - feature_min == 0, 1, feature_max - feature_min)

    M_train_norm[:, 1:3] = 2 * ((M_train[:, 1:3] - feature_min) / feature_range) - 1
    M_test_norm[:, 1:3] = 2 * ((M_test[:, 1:3] - feature_min) / feature_range) - 1

    return M_train_norm, M_test_norm
