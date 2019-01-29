import os
import pickle
import zipfile

import keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten, Dropout, BatchNormalization
from keras.models import Model
from keras.utils import to_categorical
from scipy import misc
from sklearn.preprocessing import LabelEncoder

"""
    Created by Mohsen Naghipourfar on 5/25/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""


def create_model(image_shape, n_targets, dropout_rate=0.5):
    inputs = Input(shape=image_shape, name='inputs')
    conv_1 = Conv2D(filters=96, kernel_size=(11, 11), padding='same', activation='relu', name='conv_1')(inputs)
    # conv_1 = BatchNormalization()(conv_1)
    conv_1 = Dropout(dropout_rate)(conv_1)
    max_pool_1 = MaxPooling2D((2, 2), padding='same')(conv_1)
    conv_2 = Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu', name='conv_2')(max_pool_1)
    # conv_2 = BatchNormalization()(conv_2)
    conv_2 = Dropout(dropout_rate)(conv_2)
    max_pool_2 = MaxPooling2D((2, 2), padding='same')(conv_2)
    conv_3 = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu', name='conv_3')(max_pool_2)
    # conv_3 = BatchNormalization()(conv_3)
    conv_3 = Dropout(dropout_rate)(conv_3)
    max_pool_3 = MaxPooling2D((2, 2), padding='same')(conv_3)
    conv_4 = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu', name='conv_4')(max_pool_3)
    # conv_4 = BatchNormalization()(conv_4)
    conv_4 = Dropout(dropout_rate)(conv_4)
    max_pool_4 = MaxPooling2D((2, 2), padding='same')(conv_4)
    conv_5 = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu', name='conv_5')(max_pool_4)
    # conv_5 = BatchNormalization()(conv_5)
    conv_5 = Dropout(dropout_rate)(conv_5)
    max_pool_5 = MaxPooling2D((2, 2), padding='same')(conv_5)
    flatten = Flatten()(max_pool_5)
    fc_1 = Dense(512, activation='relu')(flatten)
    # fc_1 = BatchNormalization()(fc_1)
    fc_1 = Dropout(dropout_rate)(fc_1)
    # fc_2 = Dense(64, activation='relu')(fc_1)
    # fc_2 = BatchNormalization()(fc_2)
    # fc_2 = Dropout(dropout_rate)(fc_2)
    outputs = Dense(n_targets, activation='softmax')(fc_1)

    model = Model(inputs=inputs, outputs=outputs)
    optimizer = keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_pretrained_model(x_train, y_train):
    network = keras.models.load_model('./model.hdf5')
    network.load_weights('./checkpoint.hdf5')

    checkpoint = ModelCheckpoint(filepath='./checkpoint.hdf5',
                                 verbose=0,
                                 monitor='val_acc',
                                 save_best_only=True,
                                 mode='auto',
                                 period=1)

    network.fit(x=x_train,
                y=y_train,
                epochs=200,
                batch_size=256,
                validation_split=0.2,
                callbacks=[checkpoint])

    x_test = pd.read_csv('./test.csv').as_matrix()
    x_test = x_test.reshape(28000, 28, 28, 1)
    prediction = network.predict(x=x_test)

    network.save('./model.hdf5')
    return prediction


def unzip_data(path="../Data/"):
    extracted_path = path + "extracted/"
    for i in range(1, 10):
        zip = zipfile.ZipFile(path + "Data_Part0" + str(i) + ".zip")
        zip.extractall(extracted_path)


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


def load_data(path="../Data/"):
    x_data = np.load(path + "X_train.npy")
    y_data = np.load(path + "Y_train_letters.npy")
    x_data = np.reshape(x_data, newshape=(-1, 32, 32, 1))
    return x_data, y_data


def load_test_data(path="../Data/Test/"):
    x_test = np.array(pickle.load(open(path + "x_test.pkl", 'rb'), encoding='bytes'))
    prediction = np.array(pickle.load(open(path + "prediction.pkl", 'rb'), encoding='bytes'))
    le = LabelEncoder()
    prediction = le.fit_transform(prediction)
    prediction = to_categorical(prediction)
    x_test = np.reshape(x_test, newshape=(-1, 32, 32, 1))
    return x_test, prediction


if __name__ == '__main__':
    log_path = "../results/logs/"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + "Tensorboard/", exist_ok=True)
    os.makedirs(log_path + "ModelCheckpoint/", exist_ok=True)

    # unzip_data()
    # load_and_save_data()
    x_data, y_data = load_data()

    model = create_model((32, 32, 1,), y_data.shape[1])
    model.summary()

    csv_logger = CSVLogger(log_path + "csv_log.csv")
    tensorboard_callback = TensorBoard(log_dir=log_path + "Tensorboard/", write_images=True)
    model_checkpoint = ModelCheckpoint(filepath=log_path + "ModelCheckpoint/best_model.h5", save_best_only=True,
                                       monitor="val_loss")
    x_test, y_test = load_test_data()

    model.fit(x=x_data, y=y_data,
              batch_size=512,
              epochs=200,
              verbose=2,
              callbacks=[csv_logger, tensorboard_callback, model_checkpoint],
              validation_data=(x_test, y_test))

    model.save("../results/saved_model.hdf5")
