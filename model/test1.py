import pickle

import numpy as np
from keras.models import load_model

"""
    Created by Mohsen Naghipourfar on 2019-02-03.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def load_test_data(path="../Data/Test/"):
    x_test = np.array(pickle.load(open(path + "x_test.pkl", 'rb'), encoding='bytes'))
    x_test = np.reshape(x_test, newshape=(-1, 32, 32, 1))
    x_test = np.divide(np.subtract(x_test, 127.5), 127.5)
    return x_test


x_test = load_test_data(path="../Data/")
model = load_model("../results/logs/ModelCheckpoint/best_model.h5")
y_test = model.predict(x_test)
y_test = np.array(y_test)
np.save(file="../Data/Test/prediction.pkl", arr=y_test, allow_pickle=True)
