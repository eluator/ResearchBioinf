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
from trainer import train_AgeVae, test_AgeVae
from model import VAEAge, TwoLayerNetwork

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data = pd.read_csv("./Data/Aging_data_combined_orthologs.csv")
    datawNAN = data.fillna(0)
    datawNAN = torch.Tensor(datawNAN.select_dtypes(include=['float64']).iloc[:, :-1].values).to(device)

    batch_size = 64
    validation_split = .1
    test_split = .0
    l1_lambda, l2_lambda = 0.03, 0.01
    input_size = datawNAN.shape[1] - 1
    latent_size = 10
    down_channels, up_channels = 2, 2
    hidden_size = [10, 20]
    lr = 1e-4
    epochs = 2000

    train_loader, val_loader, test_loader = split_trainValTest(datawNAN, batch_size, validation_split, test_split)
    AgeModel = TwoLayerNetwork(latent_size, l1_lambda, l2_lambda, hidden_size)
    vae = VAEAge(input_size, AgeModel, latent_size, down_channels, up_channels)
    vae_optim = Adam(vae.parameters(), lr)
    # metrics_history, metrics_history_val = train_AgeVae(vae, train_loader, val_loader, batch_size, epochs, vae_optim,
    #                                                     device)

    test_AgeVae(vae, datawNAN[:100], device, age_weight=1)

main()