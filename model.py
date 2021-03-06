import torch
from torch import nn

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

        z = mu + torch.randn_like(sigma) * torch.sqrt(sigma)
        x_pred = self._decoder(z)
        return x_pred, kld

    def encode(self, x):
        mu, log_sigma = self._encoder(x)
        sigma = torch.exp(log_sigma)

        return mu + torch.randn_like(sigma) * torch.sqrt(sigma)

    def decode(self, z):
        return self._decoder(z)


class ElasticNet(nn.Module):
    def __init__(self, input_size: int, l1_lambda: int, l2_lambda: int):
        super().__init__()
        self._input_size = input_size

        self._model = torch.nn.Linear(input_size, 1, bias=True)
        self._relu = torch.nn.ReLU()
        self._norm = torch.nn.BatchNorm1d(1)

        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        # print(self._encoder.parameters())

    def forward(self, vector):
        return self._relu(self._norm(self._model(vector))).flatten()

    def l1_reg(self):
        l1_norm = self._model.weight.abs().sum()

        return self.l1_lambda * l1_norm

    def l2_reg(self):
        l2_norm = self._model.weight.pow(2).sum()

        return self.l2_lambda * l2_norm

    def reg(self):
        return self.l1_reg() + self.l2_reg()


class TwoLayerNetwork(nn.Module):
    def __init__(self, input_size: int, l1_lambda: int, l2_lambda: int, hidden_size=[10, 20]):
        super().__init__()
        self._input_size = input_size

        self._model = nn.Sequential(torch.nn.Linear(input_size, hidden_size[0], bias=True),
                                    torch.nn.ReLU(),
                                    torch.nn.BatchNorm1d(hidden_size[0]),
                                    torch.nn.Linear(hidden_size[0], hidden_size[1], bias=True),
                                    torch.nn.ReLU(),
                                    torch.nn.BatchNorm1d(hidden_size[1]),
                                    torch.nn.Linear(hidden_size[1], 1, bias=True),
                                    torch.nn.ReLU(),
                                    torch.nn.BatchNorm1d(1))

        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        # print(self._encoder.parameters())

    def forward(self, vector):
        return self._model(vector).flatten()

    def l1_reg(self):
        l1_norm = sum([w.abs().sum() for w in self._model.parameters()])

        return self.l1_lambda * l1_norm

    def l2_reg(self):
        l2_norm = sum([w.pow(2).sum() for w in self._model.parameters()])

        return self.l2_lambda * l2_norm

    def reg(self):
        return self.l1_reg() + self.l2_reg()

class VAEAge(nn.Module):
    def __init__(self, input_size, age_model, latent_size=10, down_channels=2, up_channels=2):
        super().__init__()

        self._encoder = VaeEncoder(input_size, latent_size, down_channels)
        self._decoder = VaeDecoder(input_size, latent_size, up_channels)
        self._age = age_model

    def forward(self, x):
        mu, log_sigma = self._encoder(x)
        sigma = torch.exp(log_sigma)

        kld = 0.5 * (sigma + torch.square(mu) - log_sigma - 1)

        z = mu + torch.randn_like(sigma) * torch.sqrt(sigma)
        x_pred = self._decoder(z)
        age_pred = self._age(z)
        return x_pred, kld, age_pred

    def encode(self, x):
        mu, log_sigma = self._encoder(x)
        sigma = torch.exp(log_sigma)

        return mu + torch.randn_like(sigma) * torch.sqrt(sigma)

    def decode(self, z):
        return self._decoder(z)

    def reg(self):
        return self._age.reg()

class VAETriplets(nn.Module):
    def __init__(self, input_size, latent_size=10, down_channels=2, up_channels=2):
        super().__init__()

        self._encoder = VaeEncoder(input_size, latent_size, down_channels)
        self._decoder = VaeDecoder(input_size, latent_size, up_channels)
        self.latent_size = latent_size
        self.input_size = input_size
        self._softmax = torch.nn.Softmax(dim = 0)
        self._relu = torch.nn.ReLU()

    def forward(self, triple):
        # x_pred = torch.zeros(x.shape)
        # mu = torch.zeros([x.shape[0], x.shape[1], self.latent_size])
        # log_sigma = torch.zeros([x.shape[0], x.shape[1], self.latent_size])
        # sigma = torch.zeros([x.shape[0], x.shape[1], self.latent_size])
        # kld = torch.zeros([x.shape[1], self.latent_size])

        x, age = triple[:, :, :-1], triple[:, :, -1]

        ge_differential = self._relu((age[0] - age[1]) ** 2 - (age[1] - age[2]) ** 2).bool()

        mu_sigma = [self._encoder(xi) for i, xi in enumerate(x)]
        # print(self._encoder(xi)[0].shape, self._encoder(xi)[1].shape, xi.shape, mu[i].shape, log_sigma[i].shape)
        mu = torch.stack(tuple([x[0] for x in mu_sigma]))
        log_sigma = torch.stack(tuple([x[1] for x in mu_sigma]))
        sigma = torch.exp(log_sigma)
        kld = torch.sum(0.5 * (sigma + torch.square(mu) - log_sigma - 1), dim = 0)
        x_pred = torch.stack(tuple([self._decoder(mu[i] + torch.randn_like(sigma[i]) * torch.sqrt(sigma[i])) for i in range(3)]))

        KLij = 0.5 * (torch.log(sigma[1] / sigma[0]) + (sigma[0] + torch.square(mu[0] - mu[1])) / sigma[1] - 1)
        KLji = 0.5 * (torch.log(sigma[0] / sigma[1]) + (sigma[1] + torch.square(mu[1] - mu[0])) / sigma[0] - 1)

        KLil = 0.5 * (torch.log(sigma[2] / sigma[0]) + (sigma[0] + torch.square(mu[0] - mu[2])) / sigma[2] - 1)
        KLli = 0.5 * (torch.log(sigma[0] / sigma[2]) + (sigma[2] + torch.square(mu[2] - mu[0])) / sigma[0] - 1)

        Dij = -torch.clamp(torch.sum(KLij*0.5 + KLji*0.5, dim = 1), min=-6, max=6)
        Dil = -torch.clamp(torch.sum(KLil*0.5 + KLli*0.5, dim = 1), min=-6, max=6)
        D = torch.stack((-Dij, -Dil))

        # expDij = torch.exp(-Dij)
        pt = torch.exp(-Dij)/(torch.exp(-Dij) + torch.exp(-Dil))

        # pt = self._softmax(D)[0]
        pt_log = -torch.log(torch.cat((torch.masked_select(pt, ge_differential), (
                    1.0 - torch.masked_select(pt, torch.logical_not(ge_differential)))))).mean()

        return x_pred, kld, pt_log

    def encode(self, x):
        mu, log_sigma = self._encoder(x)
        sigma = torch.exp(log_sigma)

        return mu + torch.randn_like(sigma) * sigma

    def decode(self, z):
        return self._decoder(z)
