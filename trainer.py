import torch

def train_vae(vae, dataloader, dataloader_val, dataset, batch_size):
    vae.cuda()

    epochs = 201
    vae_optim = Adam(vae.parameters(), lr=1e-4)

    for ep in range(epochs):
        total_batches = 0
        rec_loss_avg = 0
        kld_loss_avg = 0
        r2_avg = 0

        total_batches_val = 0
        rec_loss_avg_val = 0
        kld_loss_avg_val = 0
        r2_avg_val = 0

        for i, batch in enumerate(dataloader):
            batch = batch[:, :-1]
            if len(batch) < batch_size:
                continue
            total_batches += 1
            x = batch.cuda()
            x_rec, kld = vae(x)
            kld_loss = kld.sum() / batch_size
            rec_loss = ((x_rec - x) ** 2).sum() / batch_size
            dis = ((x - torch.mean(x, dim=0)) ** 2).sum() / batch_size
            r2 = 1 - rec_loss / dis
            loss = rec_loss + 0.1 * kld_loss  # https://openreview.net/forum?id=Sy2fzU9gl
            vae_optim.zero_grad()
            loss.backward()
            vae_optim.step()
            kld_loss_avg += kld_loss.item()
            rec_loss_avg += rec_loss.item()
            r2_avg += r2.item()

        print(
            f"Epoch {ep + 1} | MSE loss: {rec_loss_avg / total_batches} | R2: {r2_avg / total_batches} | KLD loss: {kld_loss_avg / total_batches}")

        with torch.no_grad():
            for i, batch in enumerate(dataloader_val):
                batch = batch[:, :-1]
                if len(batch) < batch_size:
                    continue
                total_batches_val += 1
                x = batch.cuda()
                x_rec, kld = vae(x)
                kld_loss = kld.sum() / batch_size
                rec_loss = ((x_rec - x) ** 2).sum() / batch_size
                dis = ((x - torch.mean(x, dim=0)) ** 2).sum() / batch_size
                r2 = 1 - rec_loss / dis
                r2_avg_val += r2.item()
                kld_loss_avg_val += kld_loss.item()
                rec_loss_avg_val += rec_loss.item()

        print(
            f"Epoch {ep + 1} | MSE loss val: {rec_loss_avg_val / total_batches_val} | R2 val: {r2_avg_val / total_batches_val} | KLD loss val: {kld_loss_avg_val / total_batches_val}")


def train_AgeVae(vae, dataloader, dataloader_val, batch_size, epochs, vae_optim):
    vae.cuda()

    for ep in range(epochs):
        total_batches = 0
        rec_loss_avg = 0
        kld_loss_avg = 0
        r2_avg = 0
        age_loss_avg = 0
        age_reg_avg = 0
        r2_age_avg = 0

        total_batches_val = 0
        rec_loss_avg_val = 0
        kld_loss_avg_val = 0
        r2_avg_val = 0
        age_loss_avg_val = 0
        r2_age_avg_val = 0

        for i, batch in enumerate(dataloader):
            x, age = batch[:, :-1].cuda(), batch[:, -1].cuda()
            if len(batch) < batch_size:
                continue
            total_batches += 1
            x_rec, kld, age_pred = vae(x)
            kld_loss = kld.sum() / batch_size
            rec_loss = ((x_rec - x) ** 2).sum() / batch_size
            dis = ((x - torch.mean(x, dim=0)) ** 2).sum() / batch_size
            r2 = 1 - rec_loss / dis
            dis_age = ((age - torch.mean(age)) ** 2).sum() / len(age)
            age_loss2 = ((age_pred - age) ** 2).sum() / len(age)
            age_loss = torch.sqrt(age_loss2)
            r2_age = 1 - age_loss2 / dis_age
            if i == 0:
                print("age_pred.shape, age.shape, batch.shape, x_rec.shape, x.shape", age_pred.shape, age.shape,
                      batch.shape, x_rec.shape, x.shape)
                print("age_pred[0], age[0], batch_size, age_loss", age_pred[0], age[0], batch_size, age_loss)
            age_reg = vae.reg()
            loss = rec_loss + 0.1 * kld_loss + 500 * (age_loss + age_reg)  # https://openreview.net/forum?id=Sy2fzU9gl
            vae_optim.zero_grad()
            loss.backward()
            vae_optim.step()
            kld_loss_avg += kld_loss.item()
            rec_loss_avg += rec_loss.item()
            age_loss_avg += age_loss.item()
            age_reg_avg += age_reg.item()
            r2_age_avg += r2_age.item()
            r2_avg += r2.item()

        print(
            f"Epoch {ep + 1} | Age r2: {r2_age_avg / total_batches} | Age loss: {age_loss_avg / total_batches} |Age reg: {age_reg_avg / total_batches} | MSE loss: {rec_loss_avg / total_batches} | R2: {r2_avg / total_batches} | KLD loss: {kld_loss_avg / total_batches}")

        with torch.no_grad():
            for i, batch in enumerate(dataloader_val):
                x, age = batch[:, :-1].cuda(), batch[:, -1].cuda()
                if len(batch) < batch_size:
                    continue
                total_batches_val += 1
                x_rec, kld, age_pred = vae(x)
                kld_loss = kld.sum() / batch_size
                rec_loss = ((x_rec - x) ** 2).sum() / batch_size
                dis = ((x - torch.mean(x, dim=0)) ** 2).sum() / batch_size
                r2 = 1 - rec_loss / dis

                dis_age = ((age - torch.mean(age)) ** 2).sum() / len(age)
                age_loss2 = ((age_pred - age) ** 2).sum() / len(age)
                age_loss = torch.sqrt(age_loss2)
                r2_age = 1 - age_loss2 / dis_age

                r2_avg_val += r2.item()
                kld_loss_avg_val += kld_loss.item()
                rec_loss_avg_val += rec_loss.item()
                age_loss_avg_val += age_loss.item()
                r2_age_avg_val += r2_age.item()

        print(
            f"Epoch {ep + 1} | Age r2 val: {r2_age_avg_val / total_batches_val} | Age loss val: {age_loss_avg_val / total_batches_val} | MSE loss val: {rec_loss_avg_val / total_batches_val} | R2 val: {r2_avg_val / total_batches_val} | KLD loss val: {kld_loss_avg_val / total_batches_val}")
