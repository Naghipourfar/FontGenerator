import math
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dataset import DataLoader

"""
    Created by Mohsen Naghipourfar on 2019-01-29.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""
y_dim = 26
x_dim = 1024


def conv2d(input, name, kshape, strides=None):
    if strides is None:
        strides = [1, 1, 1, 1]
    with tf.name_scope(name):
        weights = tf.Variable(tf.truncated_normal(shape=kshape, mean=0.0, stddev=0.01), name="Weight")
        bias = tf.Variable(tf.zeros(shape=kshape[3]), name="Bias")
        out = tf.nn.conv2d(input, weights, strides=strides, padding='SAME')
        out = tf.nn.bias_add(out, bias)
        out = tf.nn.relu(out)
    return out, weights, bias


def deconv2d(input, name, kshape, n_outputs, strides=[1, 1]):
    with tf.name_scope(name):
        pass
def maxpool2d(x, name, kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.name_scope(name):
        out = tf.nn.max_pool(x,
                             ksize=kshape,
                             strides=strides,
                             padding='VALID')
        return out


class CGAN(object):
    def __init__(self, lr=0.0001, z_dim=100):
        self.lr = lr
        self.z_dim = z_dim

        self.X = tf.placeholder(tf.float32, shape=[None, x_dim], name='X')
        self.y = tf.placeholder(tf.float32, shape=[None, y_dim])
        self.Z = tf.placeholder(tf.float32, shape=[None, z_dim])
        self.create_model()

    def create_model(self):
        # Discriminator Architecture
        with tf.name_scope("Discriminator"):
            with tf.name_scope("Layer-1"):
                self.D_W1 = tf.Variable(tf.truncated_normal(shape=[x_dim + y_dim, 128], mean=0.0, stddev=0.01),
                                        name='Weight')
                self.D_b1 = tf.Variable(tf.zeros([128]), name='Bias')

            with tf.name_scope("Layer-2"):
                self.D_W2 = tf.Variable(tf.truncated_normal(shape=[128, 1], mean=0.0, stddev=0.01), name='Weight')
                self.D_b2 = tf.Variable(tf.zeros([1]), name='Bias')

            self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

        # Generator Architecture
        with tf.name_scope("Generator"):
            with tf.name_scope("DenseLayer-1"):
                self.G_W1 = tf.Variable(tf.truncated_normal(shape=[self.z_dim + y_dim, 256], mean=0.0, stddev=0.01),
                                        name='Weight')

                self.G_b1 = tf.Variable(tf.zeros([256]), name='Bias')
            with tf.name_scope("ConvLayer-1"):
                pass

            with tf.name_scope("Layer-2"):
                self.G_W2 = tf.Variable(tf.truncated_normal(shape=[128, x_dim], mean=0.0, stddev=0.01), name='Weight')
                self.G_b2 = tf.Variable(tf.zeros([x_dim]), name='Bias')

            self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        self.G_sample = self.generator(self.Z, self.y)
        D_real, D_logit_real = self.discriminator(self.X, self.y)
        D_fake, D_logit_fake = self.discriminator(self.G_sample, self.y)

        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        self.D_loss = D_loss_real + D_loss_fake
        self.G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        # Only update D(X)'s parameters, so var_list = theta_D
        self.D_solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.D_loss, var_list=self.theta_D)
        # Only update G(X)'s parameters, so var_list = theta_G
        self.G_solver = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.G_loss, var_list=self.theta_G)

    def generator(self, z, y):
        inputs = tf.concat(values=[z, y], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
        G_conv1 = deconv2d()
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)

        return G_prob

    def discriminator(self, x, y):
        inputs = tf.concat(values=[x, y], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit

    def sample_Z(self, m, n):
        """Uniform prior for G(Z)"""
        return np.random.uniform(-1., 1., size=[m, n])

    def generate_new_samples(self, sess, n_samples=16, letter='A'):
        Z_sample = self.sample_Z(n_samples, self.z_dim)

        # Create conditional one-hot vector, with index digit = 1
        y_sample = np.zeros(shape=[n_samples, y_dim])
        y_sample[:, ord(letter) - 65] = 1

        samples = sess.run(self.G_sample, feed_dict={self.Z: Z_sample, self.y: y_sample})

        samples = np.array(samples)
        samples = np.reshape(samples, newshape=(n_samples, 32, 32))

        return samples

    def fit(self, n_epochs=100, batch_size=512):
        data_loader = DataLoader(path="../Data/", batch_size=batch_size)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(n_epochs):
                for j in range(data_loader.n_samples // batch_size):
                    x_train_batch, y_train_batch = data_loader.get_next_batch()

                    _, D_loss_curr = sess.run([self.D_solver, self.D_loss],
                                              feed_dict={self.X: x_train_batch, self.y: y_train_batch,
                                                         self.Z: self.sample_Z(batch_size, self.z_dim)})
                    _, G_loss_curr = sess.run([self.G_solver, self.G_loss],
                                              feed_dict={self.y: y_train_batch,
                                                         self.Z: self.sample_Z(batch_size, self.z_dim)})
                    for j in range(14):
                        _, G_loss_curr = sess.run([self.G_solver, self.G_loss],
                                                  feed_dict={self.y: y_train_batch,
                                                             self.Z: self.sample_Z(batch_size, self.z_dim)})
                if i % 1 == 0:
                    print("Epoch: %4d\tD_loss: %.4f\tG_loss: %.4f" % (i, D_loss_curr, G_loss_curr))
                    # Generate New samples
                    n_samples, letter = 16, 'B'
                    new_samples = self.generate_new_samples(sess, n_samples=n_samples, letter=letter)
                    size = int(math.sqrt(n_samples))
                    # Display New samples
                    fig, axes = plt.subplots(size // 2, size // 2, figsize=(15, 15))

                    counter = 0
                    for row in axes:
                        for col in row:
                            col.imshow(new_samples[counter], interpolation='nearest')
                            col.axis('off')
                            counter += 1

                    os.makedirs("../results/figs/%s/" % letter, exist_ok=True)
                    plt.savefig("../results/figs/%s/new_samples_iter%d.pdf" % (letter, i))
                    plt.show()

    def dense(self, x, n_input_neurons, n_output_neurons, activation=tf.nn.relu, name_scope="Dense_1"):
        with tf.name_scope(name_scope):
            w = tf.Variable(tf.truncated_normal(shape=(n_input_neurons, n_output_neurons),
                                                mean=0, stddev=1.5, seed=1),
                            name="Weight")
            b = tf.Variable(tf.zeros([n_output_neurons]), name="Bias")
            hidden_output = tf.add(tf.matmul(x, w), b)
            activation_output = activation(hidden_output)
            # tf.summary.histogram("Weights", w)

            return activation_output


if __name__ == '__main__':
    network = CGAN(lr=0.0001, z_dim=1000)
    network.fit(n_epochs=5000,
                batch_size=512)
