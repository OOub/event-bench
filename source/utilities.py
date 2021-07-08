import math
import numpy as np
from torch.utils.data.sampler import Sampler

# specific to clusterGAN
def z_sampler(batch, z_dim , num_class = 10, label_index = None):
    if label_index is None:
        label_index = np.random.randint(low = 0 , high = num_class, size = batch)
    return np.hstack((0.10 * np.random.randn(batch, z_dim-num_class), np.eye(num_class)[label_index]))

# take a subset
def subset(dataset, subsample=100):
    data_index = np.arange(len(dataset))
    np.random.shuffle(data_index)
    if subsample > 0 and subsample < 100:
        print("Taking %s%% of the dataset" % subsample)
    
        # calculate number of samples we want to take
        data_samples = np.ceil((subsample * len(dataset)) / 100).astype(int)
        
        # choosing indices of the subset
        data_index = data_index[:data_samples]
        
    return data_index

# custom sampler for torch dataloader
class custom_sampler(Sampler):
    """Samples elements from a given list of indices.
    
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.num_samples = len(indices)
        self.indices = indices
     
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples