import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
"""
    Created by Mohsen Naghipourfar on 2019-01-29.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""

# Discriminator
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
with tf.name_scope("Discriminator"):
    with tf.name_scope("Layer-1"):
        D_W1 = tf.Variable(tf.truncated_normal(shape=[784, 128], mean=0.0, stddev=0.01), name='Weight')
        D_b1 = tf.Variable(tf.zeros([128]), name='Bias')

    with tf.name_scope("Layer-2"):
        D_W2 = tf.Variable(tf.truncated_normal(shape=[128, 1], mean=0.0, stddev=0.01), name='Weight')
        D_b2 = tf.Variable(tf.zeros([1]), name='Bias')

    theta_D = [D_W1, D_W2, D_b1, D_b2]

# Generator
Z = tf.placeholder(tf.float32, shape=[None, 100])
with tf.name_scope("Generator"):
    with tf.name_scope("Layer-1"):
        G_W1 = tf.Variable(tf.truncated_normal(shape=[100, 128], mean=0.0, stddev=0.01), name='Weight')
        G_b1 = tf.Variable(tf.zeros([128]), name='Bias')

    with tf.name_scope("Layer-2"):
        G_W2 = tf.Variable(tf.truncated_normal(shape=[128, 784], mean=0.0, stddev=0.01), name='Weight')
        G_b2 = tf.Variable(tf.zeros([784]), name='Bias')

    theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

# Only update D(X)'s parameters, so var_list = theta_D
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


def sample_Z(m, n):
    """Uniform prior for G(Z)"""
    return np.random.uniform(-1., 1., size=[m, n])


batch_size = 128
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000000):
        x_train_batch, y_train_batch = mnist.fit.next_batch(batch_size)

        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: x_train_batch, Z: sample_Z(batch_size, 100)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, 100)})
        if i % 100 == 0:
            print("Epoch: %4d\tD_loss: %.4f\tG_loss: %.4f" % (i, D_loss_curr, G_loss_curr))
