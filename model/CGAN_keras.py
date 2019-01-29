from __future__ import print_function, division

import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset import DataLoader
from keras import backend as K
from keras.layers import BatchNormalization, MaxPooling2D, LeakyReLU
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D, UpSampling2D, Conv2DTranspose, concatenate, \
    Activation
from keras.models import Model, load_model
from keras.regularizers import l1_l2
from utils import get_char_index, get_sentence_index

SAVE_RESULTS_PATH = "../results/CGAN_Adam/"


def discriminator_on_generator_loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(K.flatten(y_pred), K.ones_like(K.flatten(y_pred))), axis=-1)


def generator_l1_loss(y_true, y_pred):
    return K.mean(K.abs(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)


class CGAN(object):
    def __init__(self):
        self.width = 32
        self.height = 32
        self.channels = 1
        self.image_shape = (self.width, self.height, self.channels)
        self.num_of_letters = 26
        self.z_dim = 500
        self.discriminator = self.generator = self.CGAN = None
        self.create_network()
        # self.generator.summary()
        # self.discriminator.summary()
        # self.CGAN.summary()

    def create_network(self):
        self.generator = self.build_gen()
        adam = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        self.generator.compile(loss='binary_crossentropy', optimizer=adam)

        self.discriminator = self.build_disc()
        self.discriminator.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['mae', 'accuracy'])

        self.make_trainable(False)
        self.CGAN = self.build_CGAN()
        self.CGAN.compile(loss='binary_crossentropy', optimizer=adam, metrics=['mae'])

    def build_combined_generator(self):
        input_latent = Input(shape=(self.z_dim,), name="Z_Latent")
        input_condition = Input(shape=(self.num_of_letters,), name="Condition")
        synthetic_image = self.generator([input_latent, input_condition])

        self.discriminator.trainable = False

        discriminator_output = self.discriminator([synthetic_image, input_condition])

        self.combined = Model([input_latent, input_condition], discriminator_output)
        self.generator.compile(loss="mse", optimizer="adam")
        self.combined.compile(loss='binary_crossentropy',
                              optimizer="rmsprop")
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer="sgd",
                                   metrics=['accuracy', 'mae'])

    def build_generator(self, activation='tanh', reg=lambda: l1_l2(1e-5, 1e-5)):
        input_latent = Input((self.z_dim,), name="Gen_Z_Latent")
        input_condition = Input(shape=(self.num_of_letters,), dtype='float32', name="Gen_Condition")

        dense = concatenate([input_latent, input_condition], axis=1)
        dense = Dense(64, kernel_regularizer=reg())(dense)
        dense = LeakyReLU(0.2)(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.5)(dense)
        conv = Reshape((8, 8, 1))(dense)
        conv = Conv2DTranspose(128, (3, 3), padding="same", kernel_regularizer=reg())(conv)
        conv = LeakyReLU(0.2)(conv)
        up_sample = UpSampling2D((2, 2))(conv)
        conv = Conv2DTranspose(128, (5, 5), padding="same", kernel_regularizer=reg())(up_sample)
        conv = LeakyReLU(0.2)(conv)
        up_sample = UpSampling2D((2, 2))(conv)
        image = Conv2DTranspose(1, (11, 11), padding="same", kernel_regularizer=reg())(up_sample)
        image = Activation(activation)(image)

        generator_model = Model(inputs=[input_latent, input_condition], outputs=image, name="Generator")
        return generator_model

    def build_discriminator(self, reg=lambda: l1_l2(1e-5, 1e-5)):
        input_image = Input(shape=self.image_shape, name="Disc_input_image")
        input_condition = Input(shape=(self.num_of_letters,), dtype='float32', name="Dis_Condition")

        flat_image = Flatten()(input_image)
        dense = concatenate([flat_image, input_condition], axis=1)
        dense = Dense(np.prod(self.image_shape), kernel_regularizer=reg())(dense)
        dense = LeakyReLU(0.2)(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.25)(dense)
        dense = Reshape((self.width, self.height, self.channels))(dense)
        conv = Conv2D(64, (11, 11), padding="same", kernel_regularizer=reg())(dense)
        conv = LeakyReLU(0.2)(conv)
        max_pooling = MaxPooling2D((4, 4), padding="same")(conv)
        flat = Flatten()(max_pooling)
        dense = Dense(128, kernel_regularizer=reg())(flat)
        dense = LeakyReLU(0.2)(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.25)(dense)
        output = Dense(1, activation="sigmoid")(dense)

        discriminator_model = Model(inputs=[input_image, input_condition], outputs=output, name="Discriminator")
        # discriminator_model.summary()
        return discriminator_model

    def build_CGAN(self):
        y_real = Activation("linear")(self.discriminator(self.discriminator.inputs))
        y_fake = Activation("linear")(
            self.discriminator([self.generator(self.generator.inputs), self.discriminator.inputs[1]]))
        model = Model(self.generator.inputs + self.discriminator.inputs, [y_fake, y_real])
        return model

    def build_disc(self, reg=lambda: l1_l2(1e-5, 1e-5)):
        input_image = Input(shape=(np.prod(self.image_shape),), name="disc_image")
        input_condition = Input(shape=(self.num_of_letters,), name="disc_condition")

        inputs = concatenate([input_image, input_condition], axis=1)
        dense = Dense(1024, kernel_regularizer=reg())(inputs)
        dense = LeakyReLU(0.2)(dense)
        # dense = BatchNormalization()(dense)
        # dense = Dropout(0.5)(dense)
        dense = Dense(512, kernel_regularizer=reg())(dense)
        dense = LeakyReLU(0.2)(dense)
        # dense = BatchNormalization()(dense)
        # dense = Dropout(0.5)(dense)
        dense = Dense(256, kernel_regularizer=reg())(dense)
        dense = LeakyReLU(0.2)(dense)
        # dense = BatchNormalization()(dense)
        # dense = Dropout(0.5)(dense)
        dense = Dense(1, kernel_regularizer=reg())(dense)
        outputs = Activation('sigmoid')(dense)
        disc_model = Model([input_image, input_condition], outputs, name="disc_model")
        return disc_model

    def build_gen(self, activation='tanh', reg=lambda: l1_l2(1e-5, 1e-5)):
        input_latent = Input((self.z_dim,), name="gen_latent")
        input_condition = Input((self.num_of_letters,), name="gen_condition")

        inputs = concatenate([input_latent, input_condition], axis=1)
        dense = Dense(256, kernel_regularizer=reg())(inputs)
        dense = LeakyReLU(0.2)(dense)
        # dense = BatchNormalization()(dense)
        # dense = Dropout(0.5)(dense)
        dense = Dense(512, kernel_regularizer=reg())(dense)
        dense = LeakyReLU(0.2)(dense)
        # dense = BatchNormalization()(dense)
        # dense = Dropout(0.5)(dense)
        dense = Dense(1024, kernel_regularizer=reg())(dense)
        outputs = Activation(activation)(dense)
        gen_model = Model([input_latent, input_condition], outputs, name="gen_model")
        return gen_model

    def fit(self, epochs, batch_size=128, sample_interval=50):
        data_loader = DataLoader(path="../Data/", batch_size=batch_size)
        X_train, y_train = data_loader.X, data_loader.y
        num_batches = int(X_train.shape[0] / batch_size)
        csv_logger = pd.DataFrame(columns=['Epoch', 'D_loss', 'D_acc', 'G_loss'])
        for epoch in range(epochs + 1):
            for i in range(num_batches):
                self.make_trainable(True)
                batch_idx = np.random.randint(0, X_train.shape[0], batch_size)
                real_images, conditions = X_train[batch_idx], y_train[batch_idx]
                noise = np.random.normal(0., 1., (batch_size, self.z_dim))

                real_images = np.reshape(real_images, newshape=(-1, np.prod(self.image_shape)))
                y_real = np.ones(batch_size) + 0.2 * np.random.uniform(-1, 1, size=batch_size)
                y_fake = np.zeros(batch_size) + 0.2 * np.random.uniform(0, 1, size=batch_size)

                fake_images = self.generator.predict([noise, conditions])

                d_loss_real = self.discriminator.train_on_batch([real_images, conditions], y_real)
                d_loss_fake = self.discriminator.train_on_batch([fake_images, conditions], y_fake)
                d_loss = np.mean([d_loss_real, d_loss_fake], axis=1)

                self.make_trainable(False)
                CGAN_loss = self.CGAN.train_on_batch([noise, conditions, real_images, conditions], [y_real, y_fake])

            if epoch % 1 == 0:
                print("Epoch: %d [D_loss: %f, D_accuracy: %.2f%%] [G_loss: %f]" % (
                    epoch, d_loss[0], 100 * d_loss[1], CGAN_loss[0]))
                csv_logger = csv_logger.append(
                    {'Epoch': epoch, 'D_loss': d_loss[0], 'D_acc': 100.0 * d_loss[1], 'G_loss': CGAN_loss[0]},
                    ignore_index=True)

            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                for i in range(26):
                    char = chr(i + 65)
                    self.sample_image(char=char, epoch=epoch)
                self.sample_sent(epoch=epoch)
            csv_logger.to_csv(SAVE_RESULTS_PATH + "CSVLogger.log")

    def sample_images(self, epoch):
        r, c = 4, 7
        np.random.seed(2018 * (epoch + 15))
        noise = np.random.normal(0., 1., (r * c - 2, self.z_dim))
        sampled_labels = np.arange(0, self.num_of_letters).reshape(-1, 1)
        sampled_labels = keras.utils.to_categorical(sampled_labels, num_classes=self.num_of_letters)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        gen_imgs = np.reshape(gen_imgs, newshape=(-1, 32, 32, 1))

        gen_imgs = 127.5 * gen_imgs + 127.5

        fig, axs = plt.subplots(r, c, figsize=(40, 40))
        counter = 0
        for i in range(r):
            for j in range(c):
                if counter >= 26:
                    axs[i, j].axis('off')
                    continue
                axs[i, j].imshow(gen_imgs[counter, :, :, 0], cmap='gray')
                axs[i, j].set_title("Letter: %s" % chr(counter + 65))
                axs[i, j].axis('off')
                counter += 1
        os.makedirs(SAVE_RESULTS_PATH + "figs/", exist_ok=True)
        fig.savefig(SAVE_RESULTS_PATH + "figs/Alphabet_(Epoch=%d).pdf" % epoch)
        plt.close()

    def sample_image(self, char, epoch):
        r, c = 2, 5
        np.random.seed(2018 * (epoch + 5))
        noise = np.random.normal(0., 1., (r * c, self.z_dim))
        sampled_labels = np.array([get_char_index(char) for _ in range(r * c)]).reshape(-1, 1)
        sampled_labels = keras.utils.to_categorical(sampled_labels, num_classes=self.num_of_letters)
        fake_images = self.generator.predict([noise, sampled_labels])
        fake_images = np.reshape(fake_images, newshape=(-1, 32, 32, 1))

        fake_images = 127.5 * fake_images + 127.5

        fig, axes = plt.subplots(r, c, figsize=(20, 20))
        counter = 0
        for i in range(r):
            for j in range(c):
                axes[i, j].imshow(fake_images[counter, :, :, 0], cmap="gray")
                axes[i, j].axis("off")
        os.makedirs(SAVE_RESULTS_PATH + "figs/letters/%s/" % char, exist_ok=True)
        fig.savefig(SAVE_RESULTS_PATH + "figs/letters/%s/%d.pdf" % (char, epoch))
        plt.close()

    def sample_sentence(self, sentence="THE QUICK BROWN FOX JUMPS OVER A LAZY DOG", epoch=0):
        char_indices = get_sentence_index(sentence)
        r, c, num = 1, len(char_indices), 100
        np.random.seed(2018 * (epoch + 10))
        noise = np.random.normal(0.0, 1.0, (r * c, self.z_dim))
        sampled_labels = np.array([chr_idx for chr_idx in char_indices]).reshape(-1, 1)
        sampled_labels = keras.utils.to_categorical(sampled_labels, num_classes=self.num_of_letters)

        fake_images = self.generator.predict([noise, sampled_labels])
        fake_images = np.reshape(fake_images, newshape=(-1, 32, 32, 1))
        fake_images = 127.5 * fake_images + 127.5

        fig, axes = plt.subplots(r, c, figsize=(40, 20))
        counter = 0
        for j in range(c):
            axes[j].imshow(fake_images[counter, :, :, 0], cmap="gray")
            axes[j].axis("off")
        os.makedirs(SAVE_RESULTS_PATH + "figs/sentences/%s/" % sentence, exist_ok=True)
        fig.savefig(SAVE_RESULTS_PATH + "figs/sentences/%s/%d.pdf" % (sentence, epoch))
        plt.close()

    def sample_sent(self, sentence="THE QUICK BROWN FOX JUMPS OVER A LAZY DOG", epoch=None):
        char_indices = get_sentence_index(sentence)
        sentence_image = np.ndarray(shape=(32, 32 * 33), dtype=np.float32)

        for i, idx in enumerate(char_indices):
            fake_image = self.sample_char(chr(idx + 65))
            sentence_image[:, 32 * i:32 * (i + 1)] = fake_image[0, :, :, 0]
        plt.figure(figsize=(30, 5))
        plt.imshow(sentence_image, cmap='gray')
        plt.axis("off")
        if epoch is None:
            plt.savefig(SAVE_RESULTS_PATH + "figs/sentences/" + sentence + "/Final.pdf")
        else:
            plt.savefig(SAVE_RESULTS_PATH + "figs/sentences/" + sentence + "/%d.pdf" % epoch)

    def sample_char(self, char="A"):
        char_index = get_char_index(char)
        label = keras.utils.to_categorical([char_index], num_classes=self.num_of_letters)
        noise = np.random.normal(0.0, 1.0, size=(1, self.z_dim))

        fake_image = self.generator.predict([noise, label])
        fake_image = np.reshape(fake_image, newshape=(1, 32, 32, 1))
        return fake_image

    def make_trainable(self, value):
        self.discriminator.trainable = value
        for layer in self.discriminator.layers:
            layer.trainable = value

    def plot_training(self, log_path=SAVE_RESULTS_PATH + "CSVLogger.log"):
        log_df = pd.read_csv(log_path)
        epochs = log_df["Epoch"].values
        d_accuracies = log_df["D_acc"].values
        d_loss = log_df["D_loss"].values
        g_loss = log_df["G_loss"].values
        n_epochs = int(np.max(epochs))

        plt.figure(figsize=(15, 10))
        plt.plot(epochs, d_accuracies, label="Discriminator Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.xticks([i for i in range(0, n_epochs + 5, 5)])
        plt.yticks(np.arange(0, 105, 5).tolist())
        plt.title("Discriminator Accuracy during CGAN Training")
        plt.grid()
        plt.savefig(SAVE_RESULTS_PATH + "figs/D_Accuracy.pdf")
        plt.close("all")

        plt.figure(figsize=(15, 10))
        plt.plot(epochs, d_loss, label="Discriminator MAE Loss")
        plt.xlabel("Epochs")
        plt.ylabel("MAE Loss")
        plt.xticks([i for i in range(0, n_epochs + 5, 5)])
        # plt.yticks(np.arange(0.25, -0.05, -0.05).tolist())
        plt.title("Discriminator MAE loss during CGAN Training")
        plt.grid()
        plt.savefig(SAVE_RESULTS_PATH + "figs/D_Loss.pdf")
        plt.close("all")

        plt.figure(figsize=(15, 10))
        plt.plot(epochs, g_loss, label="Generator MAE Loss")
        plt.xlabel("Epochs")
        plt.ylabel("MAE Loss")
        plt.xticks([i for i in range(0, n_epochs + 5, 5)])
        # plt.yticks(np.arange(0.25, -0.05, -0.05).tolist())
        plt.title("Generator MAE loss during CGAN Training")
        plt.grid()
        plt.savefig(SAVE_RESULTS_PATH + "figs/G_Loss.pdf")
        plt.close("all")

    def save_model(self):
        self.generator.save(SAVE_RESULTS_PATH + "G.hdf5")
        self.discriminator.save(SAVE_RESULTS_PATH + "D.hdf5")
        self.CGAN.save(SAVE_RESULTS_PATH + "CGAN.hdf5")

    def load_model(self):
        self.generator = load_model(SAVE_RESULTS_PATH + "G.hdf5")
        self.discriminator = load_model(SAVE_RESULTS_PATH + "G.hdf5")
        self.CGAN = load_model(SAVE_RESULTS_PATH + "G.hdf5")

    def inception_score(self, X, eps=1e-20):
        kl = X * ((X + eps).log() - (X.mean(0) + eps).log().expand_as(X))
        score = np.exp(kl.sum(1).mean())
        return score

    def load_CNN_model(self, path="../results/CNN/logs/best_model.h5"):
        self.CNN = load_model(path)


if __name__ == '__main__':
    os.makedirs(SAVE_RESULTS_PATH, exist_ok=True)
    cgan = CGAN()
    # cgan.fit(epochs=100, batch_size=256, sample_interval=5)
    # cgan.save_model()
    # cgan.plot_training()
    cgan.load_model()
    for i in range(0, 100, 5):
        cgan.sample_sent(epoch=i)
