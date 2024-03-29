import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

class Discriminator(object):
    def __init__(self, x_dim = 784):
        self.x_dim = x_dim
        self.name = 'nmnist/clus_wgan/d_net'

    def __call__(self, x, reuse=True):
        with tf.compat.v1.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]

            x = tf.reshape(x, [bs, 28, 28, 1])
            conv1 = tc.layers.convolution2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv1 = leaky_relu(conv1)
            conv2 = tc.layers.convolution2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv2 = leaky_relu(conv2)
            conv2 = tcl.flatten(conv2)

            fc1 = tc.layers.fully_connected(
                conv2, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)
            fc2 = tc.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
            return fc2

    @property
    def vars(self):
        return [var for var in tf.compat.v1.global_variables() if self.name in var.name]


class Generator(object):
    def __init__(self, z_dim = 10, x_dim = 784):
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.name = 'nmnist/clus_wgan/g_net'

    def __call__(self, z):
        with tf.compat.v1.variable_scope(self.name) as vs:
            bs = tf.shape(z)[0]
            fc1 = tc.layers.fully_connected(
                z, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc1 = tc.layers.batch_norm(fc1)
            fc1 = tf.nn.relu(fc1)
            fc2 = tc.layers.fully_connected(
                fc1, 7 * 7 * 128,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc2 = tf.reshape(fc2, tf.stack([bs, 7, 7, 128]))
            fc2 = tc.layers.batch_norm(fc2)
            fc2 = tf.nn.relu(fc2)
            conv1 = tc.layers.convolution2d_transpose(
                fc2, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv1 = tc.layers.batch_norm(conv1)
            conv1 = tf.nn.relu(conv1)
            conv2 = tc.layers.convolution2d_transpose(
                conv1, 1, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.sigmoid
            )
            conv2 = tf.reshape(conv2, tf.stack([bs, self.x_dim]))
            return conv2

    @property
    def vars(self):
        return [var for var in tf.compat.v1.global_variables() if self.name in var.name]


class Encoder(object):
    def __init__(self, z_dim = 10, latent_dim = 10, x_dim = 784):
        self.z_dim = z_dim
        self.latent_dim = latent_dim
        self.x_dim = x_dim
        self.name = 'nmnist/clus_wgan/enc_net'

    def __call__(self, x, reuse=True):

        with tf.compat.v1.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs, 28, 28, 1])
            conv1 = tc.layers.convolution2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv1 = leaky_relu(conv1)
            conv2 = tc.layers.convolution2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            conv2 = leaky_relu(conv2)
            conv2 = tcl.flatten(conv2)
            fc1 = tc.layers.fully_connected(
                conv2, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                weights_regularizer=tc.layers.l2_regularizer(2.5e-5),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(fc1)
            fc2 = tc.layers.fully_connected(fc1, self.z_dim, activation_fn=tf.identity)
            logits = fc2[:, self.latent_dim:]
            y = tf.nn.softmax(logits)
            return fc2[:, 0:self.latent_dim], y, logits


    @property
    def vars(self):
        return [var for var in tf.compat.v1.global_variables() if self.name in var.name]
