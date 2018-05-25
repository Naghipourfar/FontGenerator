import numpy as np
import pandas as pd
import keras

import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Dropout, BatchNormalization, Flatten
from keras.models import Model
from keras.callbacks import History, ModelCheckpoint, CSVLogger
from sklearn.preprocessing import LabelEncoder

"""
    Created by Mohsen Naghipourfar on 5/25/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""


def create_model(image_shape, n_targets):
    inputs = Input(shape=image_shape, name='inputs')
    conv_1 = Conv2D(filters=32, strides=[3, 3], padding='same', activation='relu', name='conv_1')(inputs)
    max_pool_1 = MaxPooling2D((2, 2), padding='same')(conv_1)
    flatten = Flatten()(max_pool_1)
    fc_1 = Dense(512, activation='relu')(flatten)
    outputs = Dense(n_targets, activation='softmax')(fc_1)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


x_data = pd.read_csv("../data/fonts.csv", header=None)
y_data = pd.read_csv("../data/fonts.csv", header=None)

le = LabelEncoder()
le.fit(y_data)
le_encoded = le.transform(y_data)
y_data = keras.utils.to_categorical(y_data)

model = create_model(x_data.shape, y_data.shape[1])
model.fit(x=x_data, y=y_data,
          batch_size=256,
          epochs=200,
          verbose=2,
          callbacks=History(),
          validation_split=0.2)
