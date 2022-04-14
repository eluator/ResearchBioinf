import pandas as pd
import numpy as np
import torch
from torch.optim import Adam

import pyreadr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

from dataset import split_trainValTest
from trainer import train_triplets
from model import VAETriplets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data = pd.read_csv("./Data/Aging_data_scaled_combined_orthologs.csv")
# data.head()
datawNAN = data.fillna(0)
datawNAN = torch.Tensor(datawNAN.select_dtypes(include=['float64']).iloc[:, :-1].values)

print(datawNAN.shape)

batch_size = 20*3
validation_split = .1
test_split = .0
l1_lambda, l2_lambda = 0.03, 0.01
input_size = datawNAN.shape[1] - 1
latent_size = 10
down_channels, up_channels = 2, 2
hidden_size = [10, 20]
lr = 1e-5
epochs = 1

torch.autograd.set_detect_anomaly(True)

train_loader, val_loader, test_loader = split_trainValTest(datawNAN, batch_size, validation_split, test_split)
vae = VAETriplets(input_size, latent_size, down_channels, up_channels)
vae_optim = Adam(vae.parameters(), lr)
metrics_history, metrics_history_val = train_triplets(vae, train_loader, val_loader, batch_size, epochs, vae_optim, device)