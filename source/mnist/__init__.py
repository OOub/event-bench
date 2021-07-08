import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
mnist = input_data.read_data_sets('./data/mnist')


class DataSampler(object):
    def __init__(self, frame_time=None, subsample=None):
        self.shape = [28, 28, 1]
        self.num_classes = 10
        
    def train(self, batch_size, label=False):
        if label:
           return mnist.train.next_batch(batch_size)
        else:
           return mnist.train.next_batch(batch_size)[0]

    def test(self):
        return mnist.test.images, mnist.test.labels

    def validation(self):
        return mnist.validation.images, mnist.validation.labels


    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)

    def load_all(self):

        X_train = mnist.train.images
        X_val = mnist.validation.images
        X_test = mnist.test.images

        Y_train = mnist.train.labels
        Y_val = mnist.validation.labels
        Y_test = mnist.test.labels

        X = np.concatenate((X_train, X_val, X_test))
        Y = np.concatenate((Y_train, Y_val, Y_test))

        return X, Y.flatten()