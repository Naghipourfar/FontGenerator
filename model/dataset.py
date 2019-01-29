import numpy as np

"""
    Created by Mohsen Naghipourfar on 2019-01-29.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


class DataLoader(object):
    def __init__(self, path, batch_size):
        self.X = np.load(path + "X_train.npy")
        self.y = np.load(path + "Y_train_letters.npy")
        self.batch_size = batch_size

    def get_next_batch(self):
        """
            Return a total of `num` random samples and labels.
        """
        idx = np.arange(0, self.X.shape[0])
        np.random.shuffle(idx)
        idx = idx[:self.batch_size]
        data_shuffle = [self.X[i] for i in idx]
        labels_shuffle = [self.y[i] for i in idx]
        data_shuffle = np.asarray(data_shuffle)
        data_shuffle = np.reshape(data_shuffle, newshape=(self.batch_size, 1024))

        labels_shuffle = np.asarray(labels_shuffle)
        labels_shuffle = np.reshape(labels_shuffle, newshape=(self.batch_size, 26))
        return data_shuffle, labels_shuffle
