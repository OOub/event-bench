import tonic
import tonic.transforms as transforms
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

# initialising pokerdvs
transform = transforms.Compose([
    transforms.ToRatecodedFrame(frame_time=5000, merge_polarities=True)
])

train_set = tonic.datasets.POKERDVS(save_to='./data', train=True, download=True, transform=transform)
test_set = tonic.datasets.POKERDVS(save_to='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(train_set, shuffle=True, num_workers=2)
testloader = DataLoader(test_set, shuffle=True, num_workers=2)

iterator = iter(trainloader)
        
class DataSampler(object):
    def __init__(self):
        self.shape = [35, 35, 1]
        
    def train(self, batch_size=1, label=False):
        frames, labels = next(iterator)
        n_seq = frames.shape[1]
        frames = np.reshape(frames, (n_seq, 35**2)).numpy()
        labels = np.array([labels.item()] * n_seq)
        if label:
            return frames, labels
        else:
            return frames
    
    def test(self):
        X_test = []
        Y_test = []
        for i, (frame, label) in enumerate(testloader):
            n_seq = frame.shape[1]
            X_test.extend(np.reshape(frame, (n_seq, 35**2)).numpy())
            Y_test.extend([label.item()] * n_seq)
        X_test = np.vstack(X_test)
        Y_test = np.array(Y_test)
        return X_test, Y_test
    
    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)