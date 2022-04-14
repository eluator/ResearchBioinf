import pandas as pd
import numpy as np
from functools import partial
import os

import torch
from torch.optim import Adam

import pyreadr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

from mytune import train_tune
from model import VAEAge, TwoLayerNetwork

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


epochs = 50
num_samples = 10
max_num_epochs = epochs
gpus_per_trial = 1
cpus_per_trial = 6
data_dir = os.path.abspath("./Data/Aging_data_scaled_combined_orthologs.csv")

os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"

config = {
    "l1_lambda": tune.loguniform(1e-5, 1e-3),
    "l2_lambda": tune.loguniform(1e-5, 1e-3),
    "lr": tune.loguniform(1e-4, 1e-2),
    "hidden_size": [tune.sample_from(lambda _: 2**np.random.randint(2, 9)), tune.sample_from(lambda _: 2**np.random.randint(2, 9))],
    "latent_size": tune.choice([5, 10, 20, 50]),
    "up_channels": tune.choice([2, 3]),
    "down_channels": tune.choice([2, 3]),
    "age_weight": tune.choice([1, 10]),
    "batch_size": tune.choice([5, 10, 20])
}

scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=max_num_epochs,
    grace_period=1,
    reduction_factor=2)

reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["loss", "accuracy", "training_iteration"])

result = tune.run(
    partial(train_tune, data_dir = data_dir, epochs = epochs),
    resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter)

best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
print("Best trial final validation accuracy: {}".format(
    best_trial.last_result["accuracy"]))
