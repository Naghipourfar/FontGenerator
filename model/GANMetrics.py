import functools
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops

"""
    Created by Mohsen Naghipourfar on 2019-01-28.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""

sess = tf.InteractiveSession()
inception_images = tf.placeholder(tf.float32, [512, 1, 32, 32])


def inception_logits(images=inception_images, num_splits=1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits=num_splits)
    tfgan = tf.contrib.gan
    logits = functional_ops.map_fn(
        fn=functools.partial(tfgan.eval.run_inception, output_tensor='logits:0'),
        elems=array_ops.stack(generated_images_list),
        parallel_iterations=1,
        back_prop=False,
        swap_memory=True,
        name='RunClassifier')
    logits = array_ops.concat(array_ops.unstack(logits), 0)
    return logits


logits = inception_logits()


def get_inception_probs(inps):
    batch_size = 512
    n_batches = len(inps) // batch_size
    preds = np.zeros([n_batches * batch_size, 1000], dtype=np.float32)
    for i in range(n_batches):
        inp = inps[i * batch_size:(i + 1) * batch_size] / 255. * 2 - 1
        preds[i * batch_size:(i + 1) * batch_size] = logits.eval({inception_images: inp})[:, :1000]
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    return preds


def preds2score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def get_inception_score(images, splits=10):
    print('Calculating Inception Score with %i images in %i splits' % (images.shape[0], splits))
    start_time = time.time()
    preds = get_inception_probs(images)
    mean, std = preds2score(preds, splits)
    print('Inception Score calculation time: %f s' % (time.time() - start_time))
    return mean, std

