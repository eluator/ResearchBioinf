from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def split_trainValTest(datawNAN, batch_size, validation_split = .1, test_split = .0):
    data_size = datawNAN.shape[0]

    split_val = int(np.floor(validation_split * data_size))
    split_test = int(np.floor(test_split * data_size))

    indices = list(range(data_size))
    np.random.shuffle(indices)

    train_indices, test_indices, val_indices = indices[split_val + split_test:], indices[:split_test], \
                                               indices[:split_val + split_test]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(datawNAN, batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = DataLoader(datawNAN, batch_size=batch_size,
                                             sampler=val_sampler)
    test_loader = DataLoader(datawNAN, batch_size=batch_size,
                                              sampler=test_sampler)

    return train_loader, val_loader, test_loader