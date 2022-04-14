import torch
from graphviz import Digraph

def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            return False
        return grad_output.isnan().any() or (grad_output.abs() >= 1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                try:
                    assert fn in fn_dict, fn
                    fillcolor = 'white'
                    if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                        fillcolor = 'red'
                        print("Error red: ", fn, fn_dict[fn])
                    dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
                except:
                    print("Error green: ", fn, len(fn_dict))
                    dot.node(str(id(fn)), str(type(fn).__name__), fillcolor='green')
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot

def train_vae(vae, dataloader, dataloader_val, batch_size, epochs, vae_optim, device):
    vae.to(device)

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
            x = batch.to(device)
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
                x = batch.to(device)
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


def train_AgeVae(vae, dataloader, dataloader_val, batch_size, epochs, vae_optim, device, age_weight = 1):
    vae.to(device)
    metrics_history = []
    metrics_history_val = []

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
            x, age = batch[:, :-1].to(device), batch[:, -1].to(device)
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
            loss = rec_loss + 0.1 * kld_loss + age_weight * (age_loss + age_reg)  # https://openreview.net/forum?id=Sy2fzU9gl
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
        metrics_history.append((r2_age_avg / total_batches, r2_avg / total_batches))

        with torch.no_grad():
            for i, batch in enumerate(dataloader_val):
                x, age = batch[:, :-1].to(device), batch[:, -1].to(device)
                total_batches_val += 1
                x_rec, kld, age_pred = vae(x)
                kld_loss = kld.sum() / len(kld)
                rec_loss = ((x_rec - x) ** 2).sum() / len(x)
                dis = ((x - torch.mean(x, dim=0)) ** 2).sum() / len(x)
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
        metrics_history_val.append((r2_age_avg_val / total_batches_val, r2_avg_val / total_batches_val))

    return metrics_history, metrics_history_val

def train_regression(regr, dataloader, dataloader_val, batch_size, epochs, regr_optim, device):
    regr.to(device)
    metrics_history = []
    metrics_history_val = []

    for ep in range(epochs):
        total_batches = 0
        r2_age_avg = 0
        age_loss_avg = 0

        total_batches_val = 0
        age_loss_avg_val = 0
        r2_age_avg_val = 0

        for i, batch in enumerate(dataloader):
            x, age = torch.tensor(batch[:, :-1].to(device), requires_grad=True), torch.tensor(batch[:, -1].to(device), requires_grad=True)
            if len(batch) < batch_size:
                continue
            total_batches += 1
            age_pred = regr(x)
            dis_age = ((age - torch.mean(age)) ** 2).sum() / len(age)
            age_loss2 = ((age_pred - age) ** 2).sum() / len(age)
            age_loss = torch.sqrt(age_loss2)
            r2_age = 1 - age_loss2 / dis_age
            age_reg = regr.reg()
            loss = age_loss2 + age_reg
            regr_optim.zero_grad()
            loss.backward()
            regr_optim.step()
            age_loss_avg += age_loss.item()
            r2_age_avg += r2_age.item()

        print(
            f"Epoch {ep + 1} | sqrt MSE loss: {age_loss_avg / total_batches} | R2: {r2_age_avg / total_batches}")
        metrics_history.append(r2_age_avg / total_batches)

        with torch.no_grad():
            for i, batch in enumerate(dataloader_val):
                x, age = batch[:, :-1].to(device), batch[:, -1].to(device)
                if len(batch) < batch_size:
                    continue
                total_batches_val += 1
                age_pred = regr(x)
                dis_age = ((age - torch.mean(age)) ** 2).sum() / len(age)
                age_loss2 = ((age_pred - age) ** 2).sum() / len(age)
                age_loss = torch.sqrt(age_loss2)
                r2_age = 1 - age_loss2 / dis_age
                age_reg = regr.reg()
                loss = age_loss2 + age_reg
                age_loss_avg_val += age_loss.item()
                r2_age_avg_val += r2_age.item()

        print(
            f"Epoch {ep + 1} | sqrt MSE val loss: {age_loss_avg_val / total_batches_val} | R2 val: {r2_age_avg_val / total_batches_val}")

        metrics_history_val.append(r2_age_avg_val / total_batches_val)
    return metrics_history, metrics_history_val

def train_triplets(vae, dataloader, dataloader_val, batch_size, epochs, vae_optim, device, triplet_weight = 1):
    vae.to(device)
    metrics_history = []
    metrics_history_val = []

    for ep in range(epochs):
        total_batches = 0
        rec_loss_avg = 0
        kld_loss_avg = 0
        r2_avg = 0
        pr_loss_avg = 0
        loss_avg = 0

        total_batches_val = 0
        rec_loss_avg_val = 0
        kld_loss_avg_val = 0
        r2_avg_val = 0
        pr_loss_avg_val = 0
        loss_avg_val = 0

        for i, batch in enumerate(dataloader):
            # for x in torch.split(batch.to(device), batch_size//3):
            #     print(x.shape)
            if len(batch) < batch_size:
                continue
            x1, x2, x3 = torch.split(batch.to(device), len(batch)//3)
            triple = torch.stack((x1, x2, x3))
            total_batches += 1

            print(triple.shape)
            x_rec, kld, pt = vae(triple)
            kld_loss = kld.sum() / batch_size
            rec_loss = ((x_rec - triple[:, :, :-1]) ** 2).sum() / batch_size

            # ge = torch.ge((age[0] - age[1])**2, (age[1] - age[2])**2)
            # ge_differential = torch.nn.ReLU()((age[0] - age[1])**2 - (age[1] - age[2])**2).bool()
            # assert torch.masked_select(pt, ge).any() == torch.masked_select(pt, ge_differential).any()
            # print(torch.masked_select(pt, ge), pt, ge)

            pr_loss = triplet_weight*pt

            loss = rec_loss + 0.1 * kld_loss + pr_loss  # https://openreview.net/forum?id=Sy2fzU9gl
            vae_optim.zero_grad()
            # make_dot(loss, params=dict(vae.named_parameters()), show_attrs=True, show_saved=True)

            get_dot = register_hooks(loss)
            loss.backward(retain_graph=True)
            dot = get_dot()
            dot.save('tmp.dot') # to get .dot
            dot.render('tmp') # to get SVG
            # print(dot)

            # print(loss.grad)
            # for name, par in vae.named_parameters():
            #     print(name, par.grad)

            vae_optim.step()

            dis = (torch.stack(tuple([triple[:, :, :-1][i].detach() - torch.mean(triple[:, :, :-1][i].detach()) for i in range(3)])) ** 2).sum() / batch_size / 3.0
            r2 = 1 - rec_loss / dis
            kld_loss_avg += kld_loss.item()
            rec_loss_avg += rec_loss.item()
            r2_avg += r2.item()
            pr_loss_avg += pr_loss.item()
            loss_avg += loss.item()
        print(
            f"Epoch {ep + 1} | All loss: {loss_avg / total_batches} | MSE loss: {rec_loss_avg / total_batches} | R2: {r2_avg / total_batches} | KLD loss: {kld_loss_avg / total_batches} | Triplet loss: {pr_loss_avg / total_batches}")
        metrics_history.append((loss_avg / total_batches, r2_avg / total_batches, pr_loss_avg / total_batches))

        with torch.no_grad():
            for i, batch in enumerate(dataloader_val):
                if len(batch) < batch_size:
                    continue

                x1, x2, x3 = torch.split(batch.to(device), len(batch) // 3)
                triple = torch.stack((x1, x2, x3))
                total_batches_val += 1

                x_rec, kld, pt = vae(triple)
                kld_loss = kld.sum() / batch_size
                rec_loss = ((x_rec - triple[:, :, :-1]) ** 2).sum() / batch_size / 3.0

                # ge = torch.ge((age[0] - age[1])**2, (age[1] - age[2])**2)
                # ge_differential = torch.nn.ReLU()((age[0] - age[1])**2 - (age[1] - age[2])**2).bool()
                # assert torch.masked_select(pt, ge).any() == torch.masked_select(pt, ge_differential).any()
                # print(torch.masked_select(pt, ge), pt, ge)

                pr_loss = triplet_weight * pt

                loss = rec_loss + 0.1 * kld_loss + pr_loss  # https://openreview.net/forum?id=Sy2fzU9gl

                dis = (torch.stack(tuple([triple[:, :, :-1][i] - torch.mean(triple[:, :, :-1][i]) for i in range(3)])) ** 2).sum() / batch_size / 3.0
                r2 = 1 - rec_loss / dis
                r2_avg_val += r2.item()
                kld_loss_avg_val += kld_loss.item()
                rec_loss_avg_val += rec_loss.item()
                pr_loss_avg_val += pr_loss.item()
                loss_avg_val += loss.item()

        print(
            f"Epoch {ep + 1} | All loss: {loss_avg_val / total_batches_val} | MSE loss val: {rec_loss_avg_val / total_batches_val} | R2 val: {r2_avg_val / total_batches_val} | KLD loss: {kld_loss_avg_val / total_batches_val} | Triplet loss val: {pr_loss_avg_val / total_batches_val}")
        metrics_history.append((loss_avg_val / total_batches_val, r2_avg_val / total_batches_val, pr_loss_avg_val / total_batches_val))

    return metrics_history, metrics_history_val


def test_triplets(vae, data, device, triplet_weight = 1):
    x1, x2, x3 = torch.split(data.to(device), len(data) // 3)
    triple = torch.stack((x1, x2, x3))

    data_size = data.shape[0]
    print(triple.shape)
    x_rec, kld, pt = vae(triple)
    kld_loss = kld.sum() / data_size
    rec_loss = ((x_rec - triple[:, :, :-1]) ** 2).sum() / data_size
    x_rec, kld, pt = vae(triple)
    kld_loss = kld.sum() / batch_size
    rec_loss = ((x_rec - triple[:, :, :-1]) ** 2).sum() / data_size

    pr_loss = triplet_weight*pt
    loss = rec_loss + 0.1 * kld_loss + pr_loss  # https://openreview.net/forum?id=Sy2fzU9gl

    dis = (torch.stack(tuple([triple[:, :, :-1][i].detach() - torch.mean(triple[:, :, :-1][i].detach()) for i in range(3)])) ** 2).sum() / batch_size / 3.0
    r2 = 1 - rec_loss / dis

    print(
        f"Loss {loss} | MSE loss: {rec_loss} | R2: {r2} | KLD loss: {kld_loss} | Triplet loss: {pr_loss}")



def test_AgeVae(vae, data, device, age_weight = 1):
    vae.to(device)
    data_size = data.shape[0]

    x, age = data[:, :-1].to(device), data[:, -1].to(device)
    x_rec, kld, age_pred = vae(x)
    kld_loss = kld.sum() / data_size
    rec_loss = ((x_rec - x) ** 2).sum() / data_size
    dis = ((x - torch.mean(x, dim=0)) ** 2).sum() / data_size
    r2 = 1 - rec_loss / dis
    dis_age = ((age - torch.mean(age)) ** 2).sum() / len(age)
    age_loss2 = ((age_pred - age) ** 2).sum() / len(age)
    age_loss = torch.sqrt(age_loss2)
    r2_age = 1 - age_loss2 / dis_age
    age_reg = vae.reg()
    loss = rec_loss + 0.1 * kld_loss + age_weight * (age_loss + age_reg)  # https://openreview.net/forum?id=Sy2fzU9gl


    print(
        f"Loss {loss} | Age r2: {r2_age} | Age loss: {age_loss} |Age reg: {age_reg} | MSE loss: {rec_loss} | R2: {r2} | KLD loss: {kld_loss}")

