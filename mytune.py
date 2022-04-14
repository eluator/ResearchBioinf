import torch
from ray import tune
from torch.optim import Adam

from model import VAEAge, TwoLayerNetwork
from dataset import load_data, split_test, split_val

import os

def train_tune(config, checkpoint_dir, data_dir = None, epochs = 100, SEED = None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data, datawNAN = load_data(data_dir, restriction_spicies = "Mouse")
    train, _ = split_test(datawNAN, test_split = 0.1, SEED = SEED)
    train_loader, val_loader = split_val(train, config["batch_size"])
    input_size = datawNAN.shape[1] - 1

    AgeModel = TwoLayerNetwork(config["latent_size"], config["l1_lambda"], config["l2_lambda"], config["hidden_size"])
    vae = VAEAge(input_size, AgeModel, config["latent_size"], config["down_channels"], config["up_channels"])
    vae_optim = Adam(vae.parameters(), config["lr"])
    vae.to(device)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        vae.load_state_dict(model_state)
        vae_optim.load_state_dict(optimizer_state)

    for ep in range(epochs):
        total_batches = 0
        rec_loss_avg = 0
        kld_loss_avg = 0
        r2_avg = 0
        age_loss_avg = 0
        age_reg_avg = 0
        r2_age_avg = 0
        loss_avg = 0

        total_batches_val = 0
        rec_loss_avg_val = 0
        kld_loss_avg_val = 0
        r2_avg_val = 0
        age_loss_avg_val = 0
        r2_age_avg_val = 0
        loss_avg_val = 0

        for i, batch in enumerate(train_loader):
            x, age = batch[:, :-1].to(device), batch[:, -1].to(device)
            if len(batch) < config["batch_size"]:
                continue
            total_batches += 1
            x_rec, kld, age_pred = vae(x)
            kld_loss = kld.sum() / config["batch_size"]
            rec_loss = ((x_rec - x) ** 2).sum() / config["batch_size"]
            dis = ((x - torch.mean(x, dim=0)) ** 2).sum() / config["batch_size"]
            r2 = 1 - rec_loss / dis
            dis_age = ((age - torch.mean(age)) ** 2).sum() / len(age)
            age_loss2 = ((age_pred - age) ** 2).sum() / len(age)
            age_loss = torch.sqrt(age_loss2)
            r2_age = 1 - age_loss2 / dis_age
            if i == 0:
                print("age_pred.shape, age.shape, batch.shape, x_rec.shape, x.shape", age_pred.shape, age.shape,
                      batch.shape, x_rec.shape, x.shape)
                print("age_pred[0], age[0], batch_size, age_loss", age_pred[0], age[0], config["batch_size"], age_loss)
            age_reg = vae.reg()
            loss = rec_loss + 0.1 * kld_loss + config["age_weight"] * (
                        age_loss + age_reg)  # https://openreview.net/forum?id=Sy2fzU9gl
            vae_optim.zero_grad()
            loss.backward()
            vae_optim.step()
            kld_loss_avg += kld_loss.item()
            rec_loss_avg += rec_loss.item()
            age_loss_avg += age_loss.item()
            age_reg_avg += age_reg.item()
            r2_age_avg += r2_age.item()
            r2_avg += r2.item()
            loss_avg += loss.item()

        print(
            f"Epoch {ep + 1} | Age r2: {r2_age_avg / total_batches} | Age loss: {age_loss_avg / total_batches} |Age reg: {age_reg_avg / total_batches} | MSE loss: {rec_loss_avg / total_batches} | R2: {r2_avg / total_batches} | KLD loss: {kld_loss_avg / total_batches}")

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                x, age = batch[:, :-1].to(device), batch[:, -1].to(device)
                if len(batch) < config["batch_size"]:
                    continue
                total_batches_val += 1
                x_rec, kld, age_pred = vae(x)
                kld_loss = kld.sum() / config["batch_size"]
                rec_loss = ((x_rec - x) ** 2).sum() / config["batch_size"]
                dis = ((x - torch.mean(x, dim=0)) ** 2).sum() / config["batch_size"]
                r2 = 1 - rec_loss / dis

                dis_age = ((age - torch.mean(age)) ** 2).sum() / len(age)
                age_loss2 = ((age_pred - age) ** 2).sum() / len(age)
                age_loss = torch.sqrt(age_loss2)
                r2_age = 1 - age_loss2 / dis_age
                age_reg = vae.reg()
                loss = rec_loss + 0.1 * kld_loss + config["age_weight"] * (
                        age_loss + age_reg)  # https://openreview.net/forum?id=Sy2fzU9gl

                r2_avg_val += r2.item()
                kld_loss_avg_val += kld_loss.item()
                rec_loss_avg_val += rec_loss.item()
                age_loss_avg_val += age_loss.item()
                r2_age_avg_val += r2_age.item()
                loss_avg_val += loss.item()
        print(
            f"Epoch {ep + 1} | Age r2 val: {r2_age_avg_val / total_batches_val} | Age loss val: {age_loss_avg_val / total_batches_val} | MSE loss val: {rec_loss_avg_val / total_batches_val} | R2 val: {r2_avg_val / total_batches_val} | KLD loss val: {kld_loss_avg_val / total_batches_val}")

        with tune.checkpoint_dir(ep) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((vae.state_dict(), vae_optim.state_dict()), path)

        tune.report(loss=(loss_avg_val / total_batches_val), accuracy=r2_age_avg_val / total_batches_val)
    print("Finished Training")
