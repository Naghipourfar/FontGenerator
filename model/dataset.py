import os
import zipfile

import numpy as np
from keras.utils import to_categorical
from scipy import misc
from sklearn.preprocessing import LabelEncoder

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
        self.index_in_epoch = 0
        self.n_samples = self.X.shape[0]

        idx = np.arange(0, self.X.shape[0])
        np.random.shuffle(idx)
        self.X = self.X[idx]
        self.y = self.y[idx]
        self.X = np.reshape(self.X, newshape=(-1, 32, 32, 1))
        self.X = np.divide(np.subtract(self.X, 127.5), 127.5)

    def get_next_batch(self):
        """
            Return a total of `num` random samples and labels.
        """
        if self.index_in_epoch + self.batch_size > self.X.shape[0]:
            idx = np.arange(0, self.X.shape[0])
            self.index_in_epoch = 0
            np.random.shuffle(idx)
            self.X = self.X[idx]
            self.y = self.y[idx]
        # idx = idx[self.index_in_epoch:self.index_in_epoch + self.batch_size]
        # data_shuffle = [self.X[i] for i in idx]
        # labels_shuffle = [self.y[i] for i in idx]
        # data_shuffle = np.asarray(data_shuffle)
        # data_shuffle = np.reshape(data_shuffle, newshape=(self.batch_size, 1024))
        #
        # labels_shuffle = np.asarray(labels_shuffle)
        # labels_shuffle = np.reshape(labels_shuffle, newshape=(self.batch_size, 26))
        data_shuffle = self.X[self.index_in_epoch:self.index_in_epoch + self.batch_size, :]
        data_shuffle = np.reshape(data_shuffle, newshape=(self.batch_size, 32, 32, 1))
        labels_shuffle = self.y[self.index_in_epoch:self.index_in_epoch + self.batch_size, :]
        labels_shuffle = np.reshape(labels_shuffle, newshape=(self.batch_size, 26))
        self.index_in_epoch += self.batch_size
        return data_shuffle, labels_shuffle


def load_and_save_data(path="../Data/extracted/"):
    x_data = []
    y_data_labels = []
    y_data_letters = []
    for i in range(1, 10):
        data_path = path + "Data_Part0" + str(i) + "/"
        for image_filename in os.listdir(data_path):
            # X_data construction
            image = misc.imread(data_path + image_filename)
            x_data.append(image)

            # Y_data construction
            image_letter = image_filename.split("_")[0]
            image_label = int(image_filename.split("_")[1].split(".")[0])
            y_data_labels.append(image_label)
            y_data_letters.append(image_letter)
    x_data = np.array(x_data)
    y_data_labels = np.array(y_data_labels)
    y_data_letters = np.array(y_data_letters)

    le = LabelEncoder()
    y_data_letters = le.fit_transform(y_data_letters)

    y_data_labels = to_categorical(y_data_labels)
    y_data_letters = to_categorical(y_data_letters)

    print(x_data.shape)
    print(y_data_letters.shape)
    print(y_data_labels.shape)
    np.save(file="../Data/X_train.npy",
            arr=x_data)
    np.save(file="../Data/Y_train_labels.npy",
            arr=y_data_labels)
    np.save(file="../Data/Y_train_letters.npy",
            arr=y_data_letters)


def unzip_data(path="../Data/"):
    extracted_path = path + "extracted/"
    for i in range(1, 10):
        zip = zipfile.ZipFile(path + "Data_Part0" + str(i) + ".zip")
        zip.extractall(extracted_path)
