import tonic
import numpy as np
import tonic.transforms as transforms
from torch.utils.data import DataLoader
import sklearn.utils as sku
from ..utilities import subset, custom_sampler
import matplotlib.pyplot as plt

class DataSampler(object):
    def __init__(self, frame_time, subsample):
        self.num_classes = 10
        
        # load data
        transform = transforms.Compose([transforms.ToRatecodedFrame(frame_time=frame_time, merge_polarities=True)])
        train_set = tonic.datasets.DVSGesture(save_to='./data', train=True, download=True, transform=transform)
        test_set = tonic.datasets.DVSGesture(save_to='./data', train=False, download=True, transform=transform)

        # get subsets if applicable
        train_index = subset(train_set, subsample)
        test_index = subset(test_set, subsample)
        trainloader = DataLoader(train_set, sampler=custom_sampler(train_index), shuffle=False)
        testloader = DataLoader(test_set, sampler=custom_sampler(test_index), shuffle=False)

        # parse into one numpy array
        self.X_train, self.Y_train = parse(trainloader, shuffle=True)
        self.X_test, self.Y_test = parse(trainloader, shuffle=True)

        # dataset sizes
        self.train_size = self.X_train.shape[0]
        self.test_size = self.X_test.shape[0]

    def train(self, batch_size, label=False):
        indices = np.random.randint(low=0, high=self.train_size, size=batch_size)
        if label:
            return self.X_train[indices, :], self.Y_train[indices]
        else:
            return self.X_train[indices, :]

    def test(self):
        return self.X_test, self.Y_test

def parse(dataloader, shuffle=True):
    X = []
    Y = []
    for i, (frame, label) in enumerate(dataloader):
        print("original", frame.shape)
        plt.imshow(frame[0][0])
        plt.show()
        frame = crop(frame, target_size=(112, 112))
        print("crop", frame.shape)
        plt.imshow(frame[0][0])
        plt.show()
        frame = downsample(frame, factor=4)
        print("downsample", frame.shape)
        plt.imshow(frame[0][0])
        plt.show()
        n_seq = frame.shape[1]
        X.extend(np.reshape(frame, (n_seq, 784)))
        Y.extend([label.item()] * n_seq)

    if shuffle:
        return sku.shuffle(np.vstack(X), np.array(Y))
    else:
        return np.vstack(X), np.array(Y)

def downsample(image, factor, estimator=np.nanmean):
    """
    Downsample a 2D array by averaging over *factor* pixels in each axis.
    Crops upper edge if the shape is not a multiple of factor.
    """
    
    w = image.shape[2]
    h = image.shape[3]
    
    downsampled = np.empty((1, image.shape[1], w / factor, h / factor))
    for i in np.arange(image.shape[1]):
        cropped = image[0][i][:w-(w % int(factor)),:h-(h % int(factor))]
        downsampled[0][i] = estimator(np.concatenate([[cropped[j::factor,k::factor] for j in range(factor)] for k in range(factor)]), axis=0)
    return downsampled

def crop(image, target_size=(28,28)):
    assert(len(image.shape) == 4)

    image = image.numpy()
    w = image.shape[2]
    h = image.shape[3]
    box = (w-target_size[0])//2, (h-target_size[1])//2, (w+target_size[0])//2, (h+target_size[1])//2

    cropped = np.empty((1, image.shape[1], target_size[0], target_size[1]))
    for i in np.arange(image.shape[1]):
        cropped[0][i] = image[0][i][box[0]:box[2], box[1]:box[3]]

    return cropped
