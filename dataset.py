from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import pandas as pd
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_data(dir_data, restriction_spicies):
    data = pd.read_csv(dir_data)
    datawNAN = data.fillna(0)[data['Species'] == restriction_spicies] if restriction_spicies is not None else data.fillna(0)
    datawNAN = torch.Tensor(datawNAN.select_dtypes(include=['float64']).iloc[:, :-1].values)
    return data, datawNAN

def split_test(datawNAN, test_split = 0.1, SEED = None):
    data_size = datawNAN.shape[0]

    split_test = int(np.floor(test_split * data_size))

    indices = list(range(data_size))

    if SEED is None:
        np.random.shuffle(indices)
    else:
        rng = np.random.default_rng(SEED)
        rng.shuffle(indices)

    train_indices, test_indices = indices[split_test:], indices[:split_test]

    return datawNAN[train_indices], datawNAN[test_indices]

def split_val(datawNAN, batch_size, validation_split = .1, SEED = None):
    g = torch.Generator()
    if SEED is not None:
        g.manual_seed(SEED)

    data_size = datawNAN.shape[0]

    split_val = int(np.floor(validation_split * data_size))

    indices = list(range(data_size))
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split_val:], indices[:split_val]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(datawNAN, batch_size=batch_size,
                                               sampler=train_sampler, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(datawNAN, batch_size=batch_size,
                                             sampler=val_sampler, worker_init_fn=seed_worker, generator=g)

    return train_loader, val_loader

