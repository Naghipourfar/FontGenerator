import os
import pickle

import keras
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.models import Model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

"""
    Created by Mohsen Naghipourfar on 5/25/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""


def create_model(image_shape, n_targets, dropout_rate=0.5):
    inputs = Input(shape=image_shape, name='inputs')
    conv_1 = Conv2D(filters=96, kernel_size=(11, 11), padding='same', name='conv_1')(inputs)
    conv_1 = LeakyReLU(0.2)(conv_1)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(dropout_rate)(conv_1)
    max_pool_1 = MaxPooling2D((2, 2), padding='same')(conv_1)
    conv_2 = Conv2D(filters=256, kernel_size=(5, 5), padding='same', name='conv_2')(max_pool_1)
    conv_2 = LeakyReLU(0.2)(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Dropout(dropout_rate)(conv_2)
    max_pool_2 = MaxPooling2D((2, 2), padding='same')(conv_2)
    conv_3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same', name='conv_3')(max_pool_2)
    conv_3 = LeakyReLU(0.2)(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Dropout(dropout_rate)(conv_3)
    max_pool_3 = MaxPooling2D((2, 2), padding='same')(conv_3)
    conv_4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same', name='conv_4')(max_pool_3)
    conv_4 = LeakyReLU(0.2)(conv_4)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Dropout(dropout_rate)(conv_4)
    max_pool_4 = MaxPooling2D((2, 2), padding='same')(conv_4)
    conv_5 = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu', name='conv_5')(max_pool_4)
    conv_5 = LeakyReLU(0.2)(conv_5)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Dropout(dropout_rate)(conv_5)
    max_pool_5 = MaxPooling2D((2, 2), padding='same')(conv_5)
    flatten = Flatten()(max_pool_5)
    fc_1 = Dense(512)(flatten)
    fc_1 = LeakyReLU(0.2)(fc_1)
    fc_1 = BatchNormalization()(fc_1)
    fc_1 = Dropout(dropout_rate)(fc_1)
    fc_2 = Dense(64, activation='relu')(fc_1)
    fc_2 = BatchNormalization()(fc_2)
    fc_2 = Dropout(dropout_rate)(fc_2)
    outputs = Dense(n_targets, activation='softmax', name="outputs")(fc_1)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Nadam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_data(path="../Data/"):
    x_data = np.load(path + "X_train.npy")
    y_data = np.load(path + "Y_train_letters.npy")
    x_data = np.reshape(x_data, newshape=(-1, 32, 32, 1))

    x_data = np.divide(np.subtract(x_data, 127.5), 127.5)
    return x_data, y_data


def load_test_data(path="../Data/Test/"):
    x_test = np.array(pickle.load(open(path + "x_test.pkl", 'rb'), encoding='bytes'))
    prediction = np.array(pickle.load(open(path + "prediction.pkl", 'rb'), encoding='bytes'))
    le = LabelEncoder()
    prediction = le.fit_transform(prediction)
    prediction = to_categorical(prediction)
    x_test = np.reshape(x_test, newshape=(-1, 32, 32, 1))
    x_test = np.divide(np.subtract(x_test, 127.5), 127.5)
    return x_test, prediction


def plot_training(save_path="../results/CNN/"):
    pass


if __name__ == '__main__':
    log_path = "../results/CNN/logs/"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + "Tensorboard/", exist_ok=True)
    os.makedirs(log_path + "ModelCheckpoint/", exist_ok=True)

    # unzip_data()
    # load_and_save_data()
    x_data, y_data = load_data()

    model = create_model((32, 32, 1,), y_data.shape[1])
    model.summary()

    csv_logger = CSVLogger(log_path + "CSVLogger.csv")
    tensorboard_callback = TensorBoard(log_dir=log_path + "Tensorboard/", write_images=True)
    model_checkpoint = ModelCheckpoint(filepath=log_path + "ModelCheckpoint/weights.{epoch:02d}-{val_acc:.2f}.hdf5",
                                       save_best_only=True,
                                       monitor="val_acc")
    x_test, y_test = load_test_data()
    # model = load_model(log_path + "ModelCheckpoint/best_model.h5")

    # print(model.evaluate(x_test, y_test))
    # print(model.evaluate(x_data, y_data))
    model.fit(x=x_data, y=y_data,
              batch_size=256,
              epochs=50,
              verbose=2,
              callbacks=[csv_logger, tensorboard_callback, model_checkpoint],
              validation_data=(x_test, y_test))

    # model.save("../results/saved_model.hdf5")
