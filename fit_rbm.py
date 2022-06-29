'''
Pulled from "Learning the Stein Discrepancy for Training and 
Evaluating Energy-Based Models without Sampling" by
Grathwohl et al.
'''
import argparse
import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.optim as optim
import torchvision as tv
import torchvision.transforms as tr
from torch.utils.data import DataLoader
import os
import utils.class_utils as class_utils
import utils.func_utils as func_utils
import arrow
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--arch', type=str, default='mlp', choices=['mlp','mlp-large'])
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--rbm_hidden_dim', type=int, default=100)
parser.add_argument('--n_samples', type=int, default=1)
parser.add_argument('--burn_in', type=int, default=2000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--grad_l2', type=float, default=0.)
parser.add_argument('--l2', type=float, default=10.)
parser.add_argument('--k_iters', type=int, default=5)
parser.add_argument('--e_iters', type=int, default=1)
parser.add_argument('--report_freq', type=int, default=50)
parser.add_argument('--save_freq', type=int, default=10000)
parser.add_argument('--viz_freq', type=int, default=500)


sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
plot = lambda p, x: tv.utils.save_image(x.clamp(0, 1), p, normalize=False, nrow=sqrt(x.size(0)))


def logit(x, alpha=1e-6):
    x = x * (1 - 2 * alpha) + alpha
    return torch.log(x) - torch.log(1 - x)


def middle_transform(x):
    return x * (255. / 256.) + (torch.rand_like(x) / 256.)


def get_data(batch_size):
    transform = tr.Compose([tr.ToTensor(), middle_transform, logit])
    
    dset_train = tv.datasets.MNIST(root="data", train=True, transform=transform, download=True)
    dset_test = tv.datasets.MNIST(root="data", train=False, transform=transform, download=True)

    dload_train = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_test = DataLoader(dset_test, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    return dload_train, dload_test


def approx_jacobian_trace(fx, x):
    eps = torch.randn_like(fx)
    eps_dfdx = func_utils.keep_grad(fx, x, grad_outputs=eps)
    tr_dfdx = (eps_dfdx * eps).sum(-1)
    return tr_dfdx




if __name__ == "__main__":
    args = parser.parse_args()
    args.data_dim = 784
    args.data_shape = (1, 28, 28)
    args.save = "EBM_Training"
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    if not os.path.exists(args.save+"/figs"):
        os.mkdir(args.save+"/figs")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dload_train, dload_test = get_data(args.batch_size)

    # collect all train data from MNIST, extracting only specific digits
    full_train_data = []
    full_train_targets = []
    for _, (x, y) in enumerate(dload_train):
        full_train_data.append(x)
        full_train_targets.append(y)
    full_train_data = torch.cat(full_train_data)
    full_train_data = full_train_data.view(full_train_data.size(0), -1)
    full_train_targets = torch.cat(full_train_targets)

    train_digits_data = full_train_data
    train_digits_targets = full_train_targets

    if args.arch == 'mlp':
        net = class_utils.MLP(in_dim=args.data_dim, out_dim=1, hidden_dim=args.rbm_hidden_dim, nn_layers=2)
        critic = class_utils.MLP(in_dim=args.data_dim, out_dim=args.data_dim, hidden_dim=args.hidden_dim, nn_layers=2)
    elif args.arch == 'mlp-large':
        net = class_utils.MLP(in_dim=args.data_dim, out_dim=1, hidden_dim=args.rbm_hidden_dim, nn_layers=3)
        critic = class_utils.MLP(in_dim=args.data_dim, out_dim=args.data_dim, hidden_dim=args.hidden_dim, nn_layers=3)

    init_batch = func_utils.form_batches(full_train_data, args.batch_size)[0].view(x.size(0), -1)

    B = torch.randn((args.data_dim, args.rbm_hidden_dim)) / args.rbm_hidden_dim
    c = torch.randn((1, args.rbm_hidden_dim))
    b = init_batch.mean(0)[None, :]
    ebm = class_utils.GaussianBernoulliRBM(B, b, c, burn_in=args.burn_in)

    models = [ebm.to(device), critic.to(device)]

    init_fn = lambda: logit(torch.rand((args.batch_size, args.data_dim)))

    optimizer = optim.Adam(ebm.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, .9))
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.5, .9))

    ebm.train()
    critic.train()

    def stein_stats(distribution, x, critic):
        sq = distribution.score_function(x)
        lp = None
        fx = critic(x)
        sq_fx = (sq * fx).sum(-1)

        tr_dfdx = torch.cat([approx_jacobian_trace(fx, x)[:, None] for _ in range(args.n_samples)], dim=1).mean(dim=1)

        stats = sq_fx + tr_dfdx
        norms = (fx * fx).sum(1)
        return stats, norms, lp

    times = []
    itr = 0
    e_losses = []
    for epoch in range(args.epochs):
        losses=[]
        x_batches = func_utils.form_batches(train_digits_data, args.batch_size)
        for x in x_batches:
            start = arrow.now()

            x = x.view(x.size(0), -1)
            x = x.to(device)

            optimizer.zero_grad()
            critic_optimizer.zero_grad()

            x.requires_grad_()

            stats, norms, logp_u = stein_stats(ebm, x, critic)
            loss = stats.mean()
            l2_penalty = norms.mean() * args.l2

            cycle_iter = itr % (args.k_iters + args.e_iters)
            if cycle_iter < args.k_iters:
                (-1. * loss + l2_penalty).backward()
                critic_optimizer.step()
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()
            critic_optimizer.zero_grad()

            times.append((arrow.now()-start).total_seconds())
            losses.append(loss.item())

            if itr % args.report_freq == 0:
                print('Iter {:04d} | Time {:.4f} s | Loss {:.6f} +/- {:.6f}'.format(
                        itr, torch.tensor(times).mean(), loss.item(), stats.std()))

            if itr % args.save_freq == 0:
                for model in models:
                    model.cpu()
                torch.save({
                    'args': args,
                    'ebm_state_dict': ebm.state_dict(),
                    'critic_state_dict': critic.state_dict()
                }, os.path.join(args.save, 'checkpt.pth'))
                for model in models:
                    model.to(device)

            critic.eval()
            ebm.eval()
            if itr % args.viz_freq == 0 and itr > 0:
                p_samples = x.view(x.size(0), *args.data_shape)
                
                pp = "{}/x_p_{}_{}.png".format(args.save+"/figs", epoch, itr)
                
                plot(pp, torch.sigmoid(p_samples.cpu()))

                q_samples = ebm.sample(args.batch_size)
                q_samples = q_samples.view(q_samples.size(0), 1, 28, 28)
                pq = "{}/x_q_{}_{}.png".format(args.save+"/figs", epoch, itr)
                plot(pq, torch.sigmoid(q_samples.cpu()))

                x_c = critic(x).view(x.size(0), 1, 28, 28)
                pc = "{}/x_c_{}_{}.png".format(args.save+"/figs", epoch, itr)
                plot(pc, x_c.cpu())
            critic.train()
            ebm.train()

            itr += 1
        e_losses.append(torch.tensor(losses).mean())

    critic.eval()
    ebm.eval()
    
    p_samples = x.view(x.size(0), *args.data_shape)
    pp = "{}/x_p_FINAL.png".format(args.save+"/figs")
    plot(pp, torch.sigmoid(p_samples.cpu()))

    q_samples = ebm.sample(args.batch_size)
    q_samples = q_samples.view(q_samples.size(0), 1, 28, 28)
    pq = "{}/x_q_FINAL.png".format(args.save+"/figs")
    plot(pq, torch.sigmoid(q_samples.cpu()))

    x_c = critic(x).view(x.size(0), 1, 28, 28)
    pc = "{}/x_c_FINAL.png".format(args.save+"/figs")
    plot(pc, x_c.cpu())

    for model in models:
        model.cpu()
    torch.save({
        'args': args,
        'ebm_state_dict': ebm.state_dict(),
        'critic_state_dict': critic.state_dict()
    }, os.path.join(args.save, 'checkpt.pth'))
    for model in models:
        model.to(device)

    plt.figure(figsize=(15,5))

    plt.plot(e_losses, color='k')

    plt.xlabel("Training Epochs")
    plt.ylabel("Learned Stein Discrepancy")

    plt.savefig(args.save+"/epoch_loss.png")