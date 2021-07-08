import time
import numpy as np
import tensorflow as tf

from .utilities import z_sampler

tf.compat.v1.set_random_seed(0)

class clusterGAN(object):
    def __init__(self, generator, discriminator, encoder, data_sampler, num_classes, latent_dim, batch_size, beta_cycle_gen, beta_cycle_label):
        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder
        self.data_sampler = data_sampler
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        scale = 10.0
        self.beta_cycle_gen = beta_cycle_gen
        self.beta_cycle_label = beta_cycle_label
        
        self.x_dim = self.discriminator.x_dim
        self.z_dim = self.generator.z_dim

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.compat.v1.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.z_gen = self.z[:,0:self.latent_dim]
        self.z_hot = self.z[:,self.latent_dim:]

        self.x_ = self.generator(self.z)
        self.z_enc_gen, self.z_enc_label, self.z_enc_logits = self.encoder(self.x_, reuse=False)
        self.z_infer_gen, self.z_infer_label, self.z_infer_logits = self.encoder(self.x)

        self.d = self.discriminator(self.x, reuse=False)
        self.d_ = self.discriminator(self.x_)

        self.g_loss = tf.reduce_mean(self.d_) + \
                      self.beta_cycle_gen * tf.reduce_mean(tf.square(self.z_gen - self.z_enc_gen)) +\
                      self.beta_cycle_label * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.z_enc_logits,labels=self.z_hot))

        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

        epsilon = tf.random.uniform([], 0.0, 1.0)
        x_hat = epsilon * self.x + (1 - epsilon) * self.x_
        d_hat = self.discriminator(x_hat)

        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)

        self.d_loss = self.d_loss + ddx

        self.d_adam = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=self.discriminator.vars)
        self.g_adam = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=[self.generator.vars, self.encoder.vars]) 

        # Reconstruction Nodes
        self.recon_loss = tf.reduce_mean(tf.abs(self.x - self.x_), 1)
        self.compute_grad = tf.gradients(self.recon_loss, self.z)

        run_config = tf.compat.v1.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=run_config)

    def train(self, num_batches=500000):

        self.sess.run(tf.compat.v1.global_variables_initializer())
        start_time = time.time()
        
        for t in range(num_batches):
            
            discriminator_iters = 5
            for _ in range(0, discriminator_iters):
                x_batch = self.data_sampler.train(self.batch_size)
                z_batch = z_sampler(self.batch_size, self.z_dim, self.num_classes)
                self.sess.run(self.d_adam, feed_dict={self.x: x_batch, self.z: z_batch})

            bz = z_sampler(self.batch_size, self.z_dim, self.num_classes)
            self.sess.run(self.g_adam, feed_dict={self.z: z_batch})
            
            if (t+1) % 100 == 0:
                bx = self.x_sampler.train(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim, self.num_classes)
      

                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z: bz}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z: bz}
                )
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                      (t+1, time.time() - start_time, d_loss, g_loss))

            if (t+1) % 5000 == 0:
                bz = z_sampler(batch_size, self.z_dim, self.num_classes)
                bx = self.sess.run(self.x_, feed_dict={self.z: bz})

