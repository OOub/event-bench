import time
import numpy as np
import tensorflow as tf
import dateutil.tz
import datetime
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

from .utilities import z_sampler
from .metric import compute_purity

tf.compat.v1.set_random_seed(0)

class clusterGAN(object):
    def __init__(self, generator, discriminator, encoder, dataset, data_sampler, num_classes, latent_dim, batch_size, beta_cycle_gen, beta_cycle_label):
        self.dataset = dataset
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

        self.saver = tf.compat.v1.train.Saver()
        
        run_config = tf.compat.v1.ConfigProto()
        run_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        run_config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=run_config)

    def train(self, num_batches=500000):

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        
        self.sess.run(tf.compat.v1.global_variables_initializer())
        start_time = time.time()
        
        print('Training {} on {}, z = {} dimension, beta_n = {}, beta_c = {}'.format(self.model, self.data, self.z_dim, self.beta_cycle_gen, self.beta_cycle_label))

        im_save_dir = 'logs/{}/z{}_cyc{}_gen{}'.format(self.data, self.z_dim,self.beta_cycle_label, self.beta_cycle_gen)
        if not os.path.exists(im_save_dir):
            os.makedirs(im_save_dir)
        
        for t in range(num_batches):
            
            discriminator_iters = 5
            for _ in range(0, discriminator_iters):
                x_batch = self.data_sampler.train(self.batch_size)
                z_batch = z_sampler(self.batch_size, self.z_dim, self.num_classes)
                self.sess.run(self.d_adam, feed_dict={self.x: x_batch, self.z: z_batch})

            bz = z_sampler(self.batch_size, self.z_dim, self.num_classes)
            self.sess.run(self.g_adam, feed_dict={self.z: z_batch})
            
            if (t+1) % 100 == 0:
                x_batch = self.x_sampler.train(batch_size)
                z_batch = self.z_sampler(batch_size, self.z_dim, self.num_classes)

                d_loss = self.sess.run(self.d_loss, feed_dict={self.x: x_batch, self.z: z_batch})
                g_loss = self.sess.run(self.g_loss, feed_dict={self.z: z_batch})
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' % (t+1, time.time() - start_time, d_loss, g_loss))

            if (t+1) % 5000 == 0:
                z_batch = z_sampler(batch_size, self.z_dim, self.num_classes)
                x_batch = self.sess.run(self.x_, feed_dict={self.z: z_batch})
                
        self.recon_enc(timestamp)
        self.save(timestamp)
        
    def save(self, timestamp):

        checkpoint_dir = 'checkpoint_dir/{}/{}_{}_z{}_cyc{}_gen{}'.format(self.data, timestamp, self.model, self.z_dim, self.beta_cycle_label, self.beta_cycle_gen)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'))
        
    def load(self, pre_trained = False, timestamp = ''):
        if timestamp == '':
            print('Best Timestamp not provided. Abort !')
            checkpoint_dir = ''
        else:
            checkpoint_dir = 'checkpoint_dir/{}/{}_{}_z{}_cyc{}_gen{}'.format(self.data, timestamp, self.model, self.z_dim, self.beta_cycle_label, self.beta_cycle_gen)

        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'model.ckpt'))
        print('Restored model weights.')
        
    def recon_enc(self, timestamp):

        data_recon, label_recon = self.data_sampler.test()

        num_pts_to_plot = data_recon.shape[0]
        latent = np.zeros(shape=(num_pts_to_plot, self.z_dim))

        print('Data Shape = {}, Labels Shape = {}'.format(data_recon.shape, label_recon.shape))
        
        for b in range(int(np.ceil(num_pts_to_plot * 1.0 / self.batch_size))):
            if (b+1)*self.batch_size > num_pts_to_plot:
                pt_indx = np.arange(b*self.batch_size, num_pts_to_plot)
            else:
                pt_indx = np.arange(b*self.batch_size, (b+1)*self.batch_size)
            xtrue = data_recon[pt_indx, :]

            zhats_gen, zhats_label = self.sess.run([self.z_infer_gen, self.z_infer_label], feed_dict={self.x : xtrue})

            latent[pt_indx, :] = np.concatenate((zhats_gen, zhats_label), axis=1)


        if self.beta_cycle_gen == 0:
            self._eval_cluster(latent[:, self.dim_gen:], label_recon, timestamp)
        else:
            self._eval_cluster(latent, label_recon, timestamp)
            
    def _eval_cluster(self, latent_rep, labels_true, timestamp):
        
        km = KMeans(n_clusters=max(self.num_classes, len(np.unique(labels_true))), random_state=0).fit(latent_rep)
        labels_pred = km.labels_

        purity = compute_purity(labels_pred, labels_true)
        ari = adjusted_rand_score(labels_true, labels_pred)
        nmi = normalized_mutual_info_score(labels_true, labels_pred)

        print('Data = {}, z_dim = {}, beta_label = {}, beta_gen = {} '.format(self.data, self.z_dim, self.beta_cycle_label, self.beta_cycle_gen))
        print(' #Points = {}, K = {}, Purity = {},  NMI = {}, ARI = {},  '.format(latent_rep.shape[0], self.num_classes, purity, nmi, ari))

        with open('logs/Res_{}_{}.txt'.format(self.data, self.model), 'a+') as f:
            f.write('{}, {} : K = {}, z_dim = {}, beta_label = {}, beta_gen = {}, Purity = {}, NMI = {}, ARI = {}\n'.format(timestamp, 'Test', self.num_classes, self.z_dim, self.beta_cycle_label, self.beta_cycle_gen, purity, nmi, ari))
            f.flush()

