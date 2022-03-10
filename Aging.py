import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm.notebook import tqdm


class VaeEncoder(nn.Module):
    def __init__(self, input_size: int, latent_size: int, down_channels: int):
        super().__init__()
        self._latent_size = latent_size
        self._input_size = input_size

        in_features = input_size
        x = (input_size // (2 * latent_size)) ** (1 / down_channels)

        #         print(x, input_size, 2*latent_size, 1/down_channels)

        modules = []
        for _ in range(down_channels - 1):
            out_features = int(in_features // x)
            modules += [
                torch.nn.Linear(in_features, out_features, bias=True),
                torch.nn.BatchNorm1d(out_features),
                torch.nn.LeakyReLU()
            ]
            in_features = out_features
        modules += [torch.nn.Linear(in_features, 2 * latent_size)]
        self._encoder = nn.Sequential(*modules)
        # print(self._encoder.parameters())

    def forward(self, vector):
        encoded = self._encoder(vector)
        assert encoded.shape[1] == self._latent_size * 2
        mu, log_sigma = torch.split(encoded, self._latent_size, dim=1)
        return mu, log_sigma


class VaeDecoder(nn.Module):
    def __init__(self, output_size: int, latent_size: int, up_channels: int):
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size

        in_features = latent_size
        x = (output_size // (latent_size)) ** (1 / up_channels)

        #         print(x)
        modules = []

        for _ in range(up_channels - 1):
            out_features = int(in_features * x)
            modules += [
                torch.nn.Linear(in_features, out_features, bias=True),
                torch.nn.BatchNorm1d(out_features),
                torch.nn.LeakyReLU()
            ]
            in_features = out_features
        modules += [torch.nn.Linear(in_features, output_size)]
        self._decoder = nn.Sequential(*modules)

    def forward(self, embeddings):
        # embeddings = embeddings.reshape(*embeddings.shape, 1, 1)
        return self._decoder(embeddings)

class VAE(nn.Module):
    def __init__(self, input_size, latent_size=10, down_channels=2, up_channels=2):
        super().__init__()

        self._encoder = VaeEncoder(input_size, latent_size, down_channels)
        self._decoder = VaeDecoder(input_size, latent_size, up_channels)

    def forward(self, x):
        mu, log_sigma = self._encoder(x)
        sigma = torch.exp(log_sigma)

        kld = 0.5 * (sigma + torch.square(mu) - log_sigma - 1)

        z = mu + torch.randn_like(sigma) * sigma
        x_pred = self._decoder(z)
        return x_pred, kld

    def encode(self, x):
        mu, log_sigma = self._encoder(x)
        sigma = torch.exp(log_sigma)

        return mu + torch.randn_like(sigma) * sigma

    def decode(self, z):
        return self._decoder(z)


def train_vae(dataloader, dataset, batch_size):
    input_size = dataset.shape[1]

    vae = VAE(input_size)
    vae.cuda()

    epochs = 21
    vae_optim = Adam(vae.parameters(), lr=1e-4)

    #     test_imgs_1 = torch.cat([dataset[i].unsqueeze(0) for i in (0, 34, 76, 1509)])
    #     test_imgs_2 = torch.cat([dataset[i].unsqueeze(0) for i in (734, 123, 512, 3634)])

    for ep in range(epochs):
        total_batches = 0
        rec_loss_avg = 0
        kld_loss_avg = 0

        #         if ep % 10 == 0:
        #             with torch.no_grad():
        #                 z_1 = vae.encode(test_imgs_1.cuda())
        #                 z_2 = vae.encode(test_imgs_2.cuda())
        #                 x_int = []
        #                 for i in range(9):
        #                     z = (i * z_1 + (8 - i) * z_2) / 8
        #                     x_int.append(vae.decode(z))
        #                 x_int = torch.cat(x_int)
        #                 visualise(x_int, rows=len(test_imgs_1))
        #                 z_rand = torch.randn_like(z_1)
        #                 x_int = vae.decode(z_rand)
        #                 visualise(x_int, rows=len(test_imgs_1)//2)

        # for i, batch in tqdm(enumerate(dataloader), total=(len(dataset) + batch_size) // batch_size):
        for i, batch in enumerate(dataloader):
            if len(batch) < batch_size:
                continue
            total_batches += 1
            x = batch.cuda()
            x_rec, kld = vae(x)
            img_elems = float(np.prod(list(batch.size())))
            kld_loss = kld.sum() / batch_size
            rec_loss = ((x_rec - x) ** 2).sum() / batch_size
            loss = rec_loss + 0.1 * kld_loss  # https://openreview.net/forum?id=Sy2fzU9gl
            vae_optim.zero_grad()
            loss.backward()
            vae_optim.step()
            kld_loss_avg += kld_loss.item()
            rec_loss_avg += rec_loss.item()

        print(
            f"Epoch {ep + 1} | Reconstruction loss: {rec_loss_avg / total_batches} | KLD loss: {kld_loss_avg / total_batches}")

def main():
    data = pd.read_csv("./Data/Aging_data_combined_orthologs.csv")
    datawNAN = data.fillna(0)
    datawNAN = torch.Tensor(datawNAN.select_dtypes(include=['float64']).iloc[:, :-2].values)

    batch_size = 8

    # print(datawNAN.head())

    data_size = datawNAN.shape[0]
    validation_split = .2
    test_split = .2
    split_val = int(np.floor(validation_split * data_size))
    split_test = int(np.floor(test_split * data_size))

    indices = list(range(data_size))
    np.random.shuffle(indices)

    train_indices, val_indices, test_indices = indices[split_val + split_test:], indices[:split_val], \
                                               indices[:split_val + split_test]

    train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)
    # test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(datawNAN, batch_size=batch_size,
                                               sampler=train_sampler)

    # print("shape: ", [x for i, x in enumerate(train_loader) if i == 0])

    # val_loader = torch.utils.data.DataLoader(datawNAN, batch_size=batch_size,
    #                                          sampler=val_sampler)
    # test_loader = torch.utils.data.DataLoader(datawNAN, batch_size=batch_size,
    #                                           sampler=test_sampler)

    train_vae(train_loader, datawNAN, batch_size)

main()